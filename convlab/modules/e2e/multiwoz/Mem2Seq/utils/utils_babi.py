# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
import torch.utils.data as data
from torch.autograd import Variable
from utils.config import *


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD", EOS_token: "EOS",  SOS_token: "SOS"}
        self.n_words = 4 # Count default tokens
      
    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, src_seq, trg_seq, index_seq, gate_seq,src_word2id, trg_word2id,max_len):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.index_seqs = index_seq   
        self.gate_seq = gate_seq     
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.max_len = max_len

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        index_s = self.index_seqs[index]
        gete_s  = self.gate_seq[index]
        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)
        trg_seq = self.preprocess(trg_seq, self.trg_word2id)
        index_s = self.preprocess_inde(index_s,src_seq)
        gete_s  = self.preprocess_gate(gete_s)
        
        return src_seq, trg_seq, index_s, gete_s,self.max_len,self.src_seqs[index],self.trg_seqs[index]

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if(trg):
            sequence = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')]+ [EOS_token]
            sequence = torch.Tensor(sequence)
        else:
            sequence = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')]
            sequence = torch.Tensor(sequence)
        return sequence

    def preprocess_inde(self, sequence,src_seq):
        """Converts words to ids."""
        sequence = sequence + [len(src_seq)-1]
        sequence = torch.Tensor(sequence)
        return sequence

    def preprocess_gate(self, sequence):
        """Converts words to ids."""
        sequence = sequence + [0]
        sequence = torch.Tensor(sequence)
        return sequence

def collate_fn(data):
    def merge(sequences,max_len):
        lengths = [len(seq) for seq in sequences]
        if (max_len):
            padded_seqs = torch.ones(len(sequences), max_len[0]).long()
        else:
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate source and target sequences
    src_seqs, trg_seqs, ind_seqs, gete_s, max_len, src_plain,trg_plain = zip(*data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs,max_len)
    trg_seqs, trg_lengths = merge(trg_seqs,None)
    ind_seqs, _ = merge(ind_seqs,None)
    gete_s, _ = merge(gete_s,None)
    
    src_seqs = Variable(src_seqs).transpose(0,1)
    trg_seqs = Variable(trg_seqs).transpose(0,1)
    ind_seqs = Variable(ind_seqs).transpose(0,1)
    gete_s = Variable(gete_s).transpose(0,1)
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        ind_seqs = ind_seqs.cuda()
        gete_s = gete_s.cuda()
    return src_seqs, src_lengths, trg_seqs, trg_lengths, ind_seqs, gete_s, src_plain, trg_plain


def read_langs(file_name, max_line = None):
    logging.info(("Reading lines from {}".format(file_name)))
    # Read the file and split into lines
    data=[]
    context=""
    u=None
    r=None
    with open(file_name) as fin:
        cnt_ptr = 0
        cnt_voc = 0
        max_r_len = 0
        cnt_lin = 1
        for line in fin:
            line=line.strip()
            if line:
                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r = line.split('\t')
                    context += str(u)+" " 
                    contex_arr = context.split(' ')[LIMIT:]
                    r_index = []
                    gate = []
                    for key in r.split(' '):
                        index = [loc for loc, val in enumerate(contex_arr) if val == key]
                        if (index):
                            index = max(index)
                            gate.append(1)
                            cnt_ptr +=1
                        else: 
                            index = len(contex_arr) - 1  
                            gate.append(0)  
                            cnt_voc +=1             
                        r_index.append(index)

                    if len(r_index) > max_r_len: 
                        max_r_len = len(r_index)
                    data.append([" ".join(contex_arr)+"$$$$",r,r_index,gate])
                    context+=str(r)+" " 
                else:
                    r=line
                    if USEKB:
                        context+=str(r)+" "                    
            else:
                cnt_lin+=1
                if(max_line and cnt_lin>=max_line):
                    break
                context=""
    max_len = max([len(d[0].split(' ')) for d in data])
    avg_len = sum([len(d[0].split(' ')) for d in data]) / float(len([len(d[0].split(' ')) for d in data]))
    logging.info("Pointer percentace= {} ".format(cnt_ptr/(cnt_ptr+cnt_voc)))
    logging.info("Max responce Len: {}".format(max_r_len))
    logging.info("Max Input Len: {}".format(max_len))
    logging.info("AVG Input Len: {}".format(avg_len))
    return data, max_len, max_r_len


def get_seq(pairs,lang,batch_size,type,max_len):   
    x_seq = []
    y_seq = []
    ptr_seq = []
    gate_seq = []
    for pair in pairs:
        x_seq.append(pair[0])
        y_seq.append(pair[1])
        ptr_seq.append(pair[2])
        gate_seq.append(pair[3])
        if(type):
            lang.index_words(pair[0])
            lang.index_words(pair[1])
    
    dataset = Dataset(x_seq, y_seq,ptr_seq,gate_seq,lang.word2index, lang.word2index,max_len)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=collate_fn)
    return data_loader

def prepare_data_seq(task,batch_size=100,shuffle=True):
    file_train = 'data/dialog-bAbI-tasks/dialog-babi-task{}trn.txt'.format(task)
    file_dev = 'data/dialog-bAbI-tasks/dialog-babi-task{}dev.txt'.format(task)
    file_test = 'data/dialog-bAbI-tasks/dialog-babi-task{}tst.txt'.format(task)
    if (int(task) != 6):
        file_test_OOV = 'data/dialog-bAbI-tasks/dialog-babi-task{}tst-OOV.txt'.format(task)
    pair_train,max_len_train, max_r_train = read_langs(file_train, max_line=None)
    pair_dev,max_len_dev, max_r_dev = read_langs(file_dev, max_line=None)
    pair_test,max_len_test, max_r_test = read_langs(file_test, max_line=None)
    max_r_test_OOV = 0
    max_len_test_OOV = 0
    if (int(task) != 6):
        pair_test_OOV,max_len_test_OOV, max_r_test_OOV = read_langs(file_test_OOV, max_line=None)
    
    max_len = max(max_len_train,max_len_dev,max_len_test,max_len_test_OOV) +1
    max_r  = max(max_r_train,max_r_dev,max_r_test,max_r_test_OOV) +1
    lang = Lang()
    
    train = get_seq(pair_train,lang,batch_size,True,max_len)
    dev   = get_seq(pair_dev,lang,batch_size,False,max_len)
    test  = get_seq(pair_test,lang,batch_size,False,max_len)
    if (int(task) != 6):
        testOOV = get_seq(pair_test_OOV,lang,batch_size,False,max_len)
    else:
        testOOV = []
    
    
    logging.info("Read %s sentence pairs train" % len(pair_train))
    logging.info("Read %s sentence pairs dev" % len(pair_dev))
    logging.info("Read %s sentence pairs test" % len(pair_test))
    if (int(task) != 6):
        logging.info("Read %s sentence pairs test" % len(pair_test_OOV))    
    logging.info("Max len Input %s " % max_len)
    logging.info("Vocab_size %s " % lang.n_words)
    logging.info("USE_CUDA={}".format(USE_CUDA))
    
    return train, dev, test, testOOV, lang, max_len, max_r

