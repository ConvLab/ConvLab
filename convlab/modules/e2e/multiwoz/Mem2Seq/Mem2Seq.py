# -*- coding: utf-8 -*-

# Modified by Microsoft Corporation.
# Licensed under the MIT license.

"""
"""
import numpy as np
import torch
from nltk import word_tokenize

from .models.Mem2Seq import Mem2Seq
from .utils.config import args, USE_CUDA, UNK_token
from .utils.utils_woz_mem2seq import prepare_data_seq, generate_memory, MEM_TOKEN_SIZE


def plain2tensor(word2index, memory):
    src_seqs = []
    for token in memory:
        src_seq = []
        for word in token:
            if word in word2index:
                src_seq.append(word2index[word])
            else:
                src_seq.append(UNK_token)
        src_seqs.append([src_seq])
    return torch.LongTensor(src_seqs).cuda() if USE_CUDA else torch.LongTensor(src_seqs)

def denormalize(uttr):
	uttr = uttr.replace(' -s', 's')
	uttr = uttr.replace(' -ly', 'ly')
	uttr = uttr.replace(' -er', 'er')
	return uttr

class Mem2seq:
    def __init__(self):
        directory = args['path'].split("/")
        task = directory[-1].split('HDD')[0]
        HDD = directory[-1].split('HDD')[1].split('BSZ')[0]
        L = directory[-1].split('L')[1].split('lr')[0]
        _, _, _, _, self.lang, max_len, max_r = prepare_data_seq(task, batch_size=1)
        self.model = Mem2Seq(int(HDD),max_len,max_r,self.lang,args['path'],task, lr=0.0, n_layers=int(L), dropout=0.0, unk_mask=0)
        self.reset()
        
    def reset(self):
        self.t = 0
        self.memory = []
    
    def predict(self, query):
        usr = query
        print('Mem2Seq usr:', usr)
        #example input: 'please find a restaurant called nusha .'
        self.t += 1
        print('Mem2Seq turn:', self.t)
        usr = ' '.join(word_tokenize(usr.lower()))
        self.memory += generate_memory(usr, '$u', self.t)
        src_plain = (self.memory+[['$$$$']*MEM_TOKEN_SIZE],)
        src_seqs = plain2tensor(self.lang.word2index, src_plain[0])
        words = self.model.evaluate_batch(1, src_seqs, [len(src_plain[0])], None, None, None, None, src_plain)
        row = np.transpose(words)[0].tolist()
        if '<EOS>' in row:
            row = row[:row.index('<EOS>')]
        sys = ' '.join(row)
        sys = denormalize(sys)
        print('Mem2Seq sys:', sys)
        self.memory += generate_memory(sys, '$s', self.t)
        return sys
        
