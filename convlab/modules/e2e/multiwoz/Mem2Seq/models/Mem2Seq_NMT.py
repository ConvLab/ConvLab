# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils.config import *
from utils.masked_cross_entropy import *
from utils.measures import wer, moses_multi_bleu


class Mem2Seq(nn.Module):
    def __init__(self, hidden_size, max_len, max_r, lang, path, lr, n_layers, dropout):
        super(Mem2Seq, self).__init__()
        self.name = "Mem2Seq"
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len ## max input
        self.max_r = max_r ## max responce len        
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        
        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
        else:
            self.encoder = EncoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout)
            self.decoder = DecoderrMemNN(lang.n_words, hidden_size, n_layers, self.dropout)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer,mode='max',factor=0.5,patience=1,min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_gate = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr =  self.loss_ptr / self.print_every
        print_loss_vac =  self.loss_vac / self.print_every
        self.print_every += 1     
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg,print_loss_vac,print_loss_ptr)
    
    def save_model(self, dec_type):
        directory = 'save/mem2seq_'+'HDD'+str(self.hidden_size)+'BSZ'+str(self.batch_size)+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type)                 
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory+'/enc.th')
        torch.save(self.decoder, directory+'/dec.th')
        
    def train_batch(self, input_batches, input_lengths, target_batches, 
                    target_lengths, target_index, batch_size, clip,
                    teacher_forcing_ratio,reset):  
        if reset:
            self.loss = 0
            self.loss_gate = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1
            
        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab,loss_Ptr= 0,0

        # Run words through encoder
        decoder_hidden = self.encoder(input_batches.transpose(0,1)).unsqueeze(0)

        # load memories with input
        self.decoder.load_memory(input_batches.transpose(0,1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        
        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(max_target_length, batch_size, input_batches.size(0)))

        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            decoder_input = decoder_input.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        if use_teacher_forcing:    
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden  = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                decoder_input = target_batches[t]# Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()            
        else:
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                _, toppi = decoder_ptr.data.topk(1)
                _, topvi = decoder_vacab.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                ## get the correspective word in input
                top_ptr_i = torch.gather(input_batches,0,Variable(toppi.view(1, -1)))
                next_in = [top_ptr_i.squeeze()[i].data[0] if(toppi.squeeze()[i] < input_lengths[i]-1) else topvi.squeeze()[i] for i in range(batch_size)]
                decoder_input = Variable(torch.LongTensor(next_in)) # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
                  
        #Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(), # -> batch x seq
            target_index.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab + loss_Ptr
        loss.backward()
        
        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.data[0]
        #self.loss_gate += loss_gate.data[0] 
        self.loss_ptr += loss_Ptr.data[0]
        self.loss_vac += loss_Vocab.data[0]
        
    def evaluate_batch(self,batch_size,input_batches, input_lengths, target_batches, target_lengths, target_index,src_plain):  
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)  
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches.transpose(0,1)).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0,1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(self.max_r, batch_size, input_batches.size(0)))
        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            decoder_input = decoder_input.cuda()
        
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_ptr,decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
            all_decoder_outputs_vocab[t] = decoder_vacab
            _, topvi = decoder_vacab.data.topk(1)
            all_decoder_outputs_ptr[t] = decoder_ptr
            _, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches,0,Variable(toppi.view(1, -1)))      
            next_in = [top_ptr_i.squeeze()[i].data[0] if(toppi.squeeze()[i] < input_lengths[i]-1) else topvi.squeeze()[i] for i in range(batch_size)]

            decoder_input = Variable(torch.LongTensor(next_in)) # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            temp = []
            for i in range(batch_size):
                if(toppi.squeeze()[i] < len(src_plain[i])-1 ):
                    temp.append(src_plain[i][toppi.squeeze()[i]]) ## copy from the input
                else:
                    ind = topvi.squeeze()[i]
                    if ind == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(self.lang.index2word[ind]) ## get from vocabulary
            decoded_words.append(temp)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words


    def evaluate(self,dev,avg_best,BLEU=False):
        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        ref = []
        hyp = []
        pbar = tqdm(enumerate(dev),total=len(dev))
        for j, data_dev in pbar: 
            words = self.evaluate_batch(
                                        batch_size=len(data_dev[1]),
                                        input_batches=data_dev[0], 
                                        input_lengths=data_dev[1], 
                                        target_batches=data_dev[2], 
                                        target_lengths=data_dev[3], 
                                        target_index=data_dev[4],
                                        src_plain=data_dev[5])
            acc=0
            w = 0 
            temp_gen = []
            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e== '<EOS>': break
                    else: st+= e + ' '
                temp_gen.append(st)
                correct = " ".join(data_dev[6][i])
                ### IMPORTANT 
                ### WE NEED TO COMPARE THE PLAIN STRING, BECAUSE WE COPY THE WORDS FROM THE INPUT 
                ### ====>> the index in the output gold can be UNK 
                if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                    acc+=1
                w += wer(correct.lstrip().rstrip(),st.lstrip().rstrip())
                ref.append(str(correct.lstrip().rstrip()))
                hyp.append(str(st.lstrip().rstrip()))

            acc_avg += acc/float(len(data_dev[1]))
            wer_avg += w/float(len(data_dev[1]))            
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg/float(len(dev)),wer_avg/float(len(dev))))

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True) 
        logging.info("BLEU SCORE:"+str(bleu_score))     
                                                             
        if (bleu_score >= avg_best):
            self.save_model(str(self.name)+str(bleu_score))
            logging.info("MODEL SAVED")  
        return bleu_score



class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        
    def get_state(self,bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(bsz, self.embedding_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.embedding_dim))


    def forward(self, story):
        u = [self.get_state(story.size(0))]
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1).long()) # b * (m * s) * e
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob   = self.softmax(torch.sum(embed_A*u_temp, 2))  
            embed_C = self.C[hop+1](story.contiguous().view(story.size(0), -1).long())
            prob = prob.unsqueeze(2).expand_as(embed_C)
            o_k  = torch.sum(embed_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)   
        return u_k

class DecoderrMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(DecoderrMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.W = nn.Linear(embedding_dim,1)
        self.W1 = nn.Linear(2*embedding_dim,self.num_vocab)
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)

    def load_memory(self, story):
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1))#.long()) # b * (m * s) * e
            m_A = embed_A    
            embed_C = self.C[hop+1](story.contiguous().view(story.size(0), -1).long())
            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    def ptrMemDecoder(self, enc_query, last_hidden):
        embed_q = self.C[0](enc_query) # b * e
        _, hidden = self.gru(embed_q.unsqueeze(0), last_hidden)
        temp = []
        u = [hidden[0].squeeze()]   
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A*u_temp, 2)
            prob_   = self.softmax(prob_lg)
            m_C = self.m_story[hop+1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            if (hop==0):
                p_vocab = self.W1(torch.cat((u[0], o_k),1))
            u_k = u[-1] + o_k
            u.append(u_k)
        p_ptr = prob_lg 
        return p_ptr, p_vocab, hidden


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
