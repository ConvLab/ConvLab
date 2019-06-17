# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from utils.config import *
from utils.masked_cross_entropy import *
from utils.measures import wer, moses_multi_bleu


class VanillaSeqToSeq(nn.Module):
    def __init__(self,hidden_size,max_len,max_r,lang,path,task,lr=0.01,n_layers=1, dropout=0.1):
        super(VanillaSeqToSeq, self).__init__()
        self.name = "VanillaSeqToSeq"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len ## max input
        self.max_r = max_r ## max responce len   
        self.lang = lang
        self.lr = lr
        self.decoder_learning_ratio = 1.0
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
                self.decoder.viz_arr =[] 
        else:
            self.encoder = EncoderRNN(lang.n_words, hidden_size, n_layers,dropout)
            self.decoder = VanillaDecoderRNN(hidden_size, lang.n_words, self.max_len, n_layers, dropout)

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr * self.decoder_learning_ratio)
        
        self.loss = 0
        self.print_every = 1
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        self.print_every += 1
        return 'L:{:.2f}'.format(print_loss_avg)

    def save_model(self,dec_type):
        name_data = "KVR/" if self.task=='' else "BABI/"
        if USEKB:
            directory = 'save/vanilla_KB-'+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type)         
        else:
            directory = 'save/vanilla_noKB-'+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'DR'+str(self.dropout)+'L'+str(self.n_layers)+'lr'+str(self.lr)+str(dec_type)         
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory+'/enc.th')
        torch.save(self.decoder, directory+'/dec.th')
        
    def load_model(self,file_name_enc,file_name_dec):
        self.encoder = torch.load(file_name_enc)
        self.decoder = torch.load(file_name_dec)


    def train_batch(self, input_batches, input_lengths, target_batches, 
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio, reset):    
        # Zero gradients of both optimizers
        if reset:
            self.loss = 0
            self.print_every = 1

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # self.opt.zero_grad()
        loss_Vocab,loss_Ptr,loss_Gate = 0,0,0
        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths)
      
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers],encoder_hidden[1][:self.decoder.n_layers])

        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, batch_size, self.output_size))
        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            decoder_input = decoder_input.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        if use_teacher_forcing:    
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_vacab, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

                all_decoder_outputs_vocab[t] = decoder_vacab
                decoder_input = target_batches[t] # Next input is current target
                if USE_CUDA: decoder_input = decoder_input.cuda()
                
        else:
            for t in range(max_target_length):
                decoder_vacab,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                all_decoder_outputs_vocab[t] = decoder_vacab
                topv, topi = decoder_vacab.data.topk(1)
                decoder_input = Variable(topi.view(-1)) # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
                  
        #Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), # -> batch x seq
            target_batches.transpose(0, 1).contiguous(), # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab
        loss.backward()
        
        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # self.opt.step()
        
        self.loss += loss.data[0]
        



    def evaluate_batch(self,batch_size,input_batches, input_lengths, target_batches):  
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)  
        # Run words through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers],encoder_hidden[1][:self.decoder.n_layers])

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.decoder.output_size))
        # Move new Variables to CUDA

        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            decoder_input = decoder_input.cuda()
        
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_vacab,decoder_hidden  = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            all_decoder_outputs_vocab[t] = decoder_vacab
            topv, topi = decoder_vacab.data.topk(1)
            decoder_input = Variable(topi.view(-1))
    
            decoded_words.append(['<EOS>'if ni == EOS_token else self.lang.index2word[ni] for ni in topi.view(-1)])
            # Next input is chosen word
            if USE_CUDA: decoder_input = decoder_input.cuda()

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)
        
        return decoded_words

    def evaluate(self,dev,avg_best,BLEU=False):
        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        pbar = tqdm(enumerate(dev),total=len(dev))
        for j, data_dev in pbar: 
            words = self.evaluate_batch(len(data_dev[1]),data_dev[0],data_dev[1],data_dev[2])             
            acc=0
            w = 0
            temp_gen = []
            #print(words)
            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e== '<EOS>':
                        break
                    else:
                        st+= e + ' '
                temp_gen.append(st)
                correct = data_dev[7][i]  

                if (correct.lstrip().rstrip() == st.lstrip().rstrip()):
                    acc+=1
                #else:
                #    print("Correct:"+str(correct.lstrip().rstrip()))
                #    print("\tPredict:"+str(st.lstrip().rstrip()))
                #    print("\tFrom:"+str(self.from_whichs[:,i]))

                w += wer(correct.lstrip().rstrip(),st.lstrip().rstrip())
                ref.append(str(correct.lstrip().rstrip()))
                hyp.append(str(st.lstrip().rstrip()))
                ref_s+=str(correct.lstrip().rstrip())+ "\n"
                hyp_s+=str(st.lstrip().rstrip()) + "\n"

            acc_avg += acc/float(len(data_dev[1]))
            wer_avg += w/float(len(data_dev[1]))
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg/float(len(dev)),
                                                                    wer_avg/float(len(dev))))

        if (BLEU):       
            bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True) 
            logging.info("BLEU SCORE:"+str(bleu_score))     
                                                                      
            if (bleu_score >= avg_best):
                self.save_model(str(self.name)+str(bleu_score))
                logging.info("MODEL SAVED")
            return bleu_score
        else:
            acc_avg = acc_avg/float(len(dev))
            if (acc_avg >= avg_best):
                self.save_model(str(self.name)+str(acc_avg))
                logging.info("MODEL SAVED")
            return acc_avg


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()      
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout      
        self.embedding_dropout = nn.Dropout(dropout) 
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout)
        if USE_CUDA:
            self.lstm = self.lstm.cuda() 
            self.embedding_dropout = self.embedding_dropout.cuda()
            self.embedding = self.embedding.cuda() 

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(1)
        h0_encoder = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size )) ### * self.num_directions = 2 if bi
        c0_encoder = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size ))  
        if USE_CUDA:
            h0_encoder = h0_encoder.cuda()
            c0_encoder = c0_encoder.cuda() 
        return h0_encoder, c0_encoder

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.embedding_dropout(embedded)        
        h0_encoder, c0_encoder = self.get_state(input_seqs)
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, (src_h_t, src_c_t) = self.lstm(embedded, (h0_encoder, c0_encoder))
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        return outputs, (src_h_t, src_c_t)

class VanillaDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_len, n_layers=1, dropout=0.1):
        super(VanillaDecoderRNN, self).__init__()
        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        if USE_CUDA:
            self.embedding = self.embedding.cuda()
            self.embedding_dropout = self.embedding_dropout.cuda()
            self.lstm = self.lstm.cuda()
            self.out = self.out.cuda()

    def forward(self, input_seq, last_hidden, encoder_outputs):
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N
        rnn_output, hidden = self.lstm(embedded, last_hidden)
        output = self.out(rnn_output)

        return output.squeeze(0),hidden
