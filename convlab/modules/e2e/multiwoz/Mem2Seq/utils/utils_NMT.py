# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
import torch.utils.data as data
from nltk.tokenize import word_tokenize
from torch.autograd import Variable
from utils.config import *


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD", EOS_token: "EOS",  SOS_token: "SOS"}
        self.n_words = 4 # Count default tokens
      
    def index_words(self, story):
        for word in story:
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
    def __init__(self, src_seq, trg_seq, index_seq ,src_word2id, trg_word2id,max_len):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.index_seqs = index_seq       
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.max_len = max_len

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.preprocess(self.src_seqs[index], self.src_word2id, trg=False)
        trg_seq = self.preprocess(self.trg_seqs[index], self.trg_word2id)
        index_s = self.preprocess_inde(self.index_seqs[index],src_seq)  
        return src_seq, trg_seq, index_s, self.max_len, self.src_seqs[index],self.trg_seqs[index]

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence]+ [EOS_token]
        else:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence]
        
        story = torch.Tensor(story)

        return story

    def preprocess_inde(self, sequence, src_seq):
        """Converts words to ids."""
        sequence = sequence + [len(src_seq)-1]
        sequence = torch.Tensor(sequence)
        return sequence

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[-1]), reverse=True)
    # seperate source and target sequences
    src_seqs, trg_seqs, ind_seqs, max_len, src_plain,trg_plain = zip(*data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    ind_seqs, _ = merge(ind_seqs)
    
    src_seqs = Variable(src_seqs).transpose(0,1)
    trg_seqs = Variable(trg_seqs).transpose(0,1)
    ind_seqs = Variable(ind_seqs).transpose(0,1)

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        ind_seqs = ind_seqs.cuda()
    return src_seqs, src_lengths, trg_seqs, trg_lengths, ind_seqs, src_plain, trg_plain


    
def read_langs(file_name, max_line = None):
    logging.info(("Reading lines from {}".format(file_name)))
    data=[]

    with open(file_name) as fin:
        cnt_ptr = 0
        cnt_voc = 0
        max_r_len = 0
        for line in fin:
            line=line.strip()
            if line:
                eng, fre = line.split('\t')
                eng, fre = word_tokenize(eng.lower()), word_tokenize(fre.lower())     
                ptr_index = []
                for key in fre:
                    index = [loc for loc, val in enumerate(eng) if (val[0] == key)]
                    if (index):
                        index = max(index)
                        cnt_ptr +=1
                    else: 
                        index = len(eng) ## sentinel 
                        cnt_voc +=1             
                    ptr_index.append(index)

                if len(ptr_index) > max_r_len: 
                    max_r_len = len(ptr_index)
                eng = eng + ['$$$$']       
                # print(eng,fre,ptr_index)
                data.append([eng,fre,ptr_index])


    max_len = max([len(d[0]) for d in data])
    logging.info("Pointer percentace= {} ".format(cnt_ptr/(cnt_ptr+cnt_voc)))
    logging.info("Max responce Len: {}".format(max_r_len))
    logging.info("Max Input Len: {}".format(max_len))
    logging.info('Sample: Eng = {}, Fre = {}, Ptr = {}'.format(" ".join(data[0][0])," ".join(data[0][1]),data[0][2]))
    return data, max_len, max_r_len


def get_seq(pairs,lang,batch_size,type,max_len):   
    x_seq = []
    y_seq = []
    ptr_seq = []
    for pair in pairs:
        x_seq.append(pair[0])
        y_seq.append(pair[1])
        ptr_seq.append(pair[2])
        if(type):
            lang.index_words(pair[0])
            lang.index_words(pair[1])
    
    dataset = Dataset(x_seq, y_seq,ptr_seq,lang.word2index, lang.word2index,max_len)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=collate_fn)
    return data_loader

def prepare_data_seq(batch_size=100,shuffle=True):
    file_train = 'data/eng-fra.txt'

    pair_train,max_len, max_r = read_langs(file_train,max_line=None)

    lang = Lang()
    
    train = get_seq(pair_train,lang,batch_size,True,max_len)
    
    logging.info("Read %s sentence pairs train" % len(pair_train))
 
    logging.info("Max len Input %s " % max_len)
    logging.info("Vocab_size %s " % lang.n_words)
    logging.info("USE_CUDA={}".format(USE_CUDA))

    return train,lang, max_len, max_r


