# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:56:40 2019

@author: truthless
"""
import json
import torch
import os
import zipfile
import pickle
from convlab.modules.word_policy.multiwoz.hdsa.transformer.Transformer import TableSemanticDecoder
from convlab.modules.word_policy.multiwoz.hdsa.transformer import Constants
from convlab.lib.file_util import cached_path

class Tokenizer(object):
    def __init__(self, vocab, ivocab, use_field, lower_case=True):
        super(Tokenizer, self).__init__()
        self.lower_case = lower_case
        self.ivocab = ivocab
        self.vocab = vocab
        self.use_field = use_field
        if use_field:
            with open('data/placeholder.json') as f:
                self.fields = json.load(f)['field']
        
        self.vocab_len = len(self.vocab)

    def tokenize(self, sent):
        if self.lower_case:
            return sent.lower().split()
        else:
            return sent.split()

    def get_word_id(self, w, template=None):
        if self.use_field and template:
            if w in self.fields and w in template:
                return template.index(w) + self.vocab_len
        
        if w in self.vocab:
            return self.vocab[w]
        else:
            return self.vocab[Constants.UNK_WORD]
        
    
    def get_word(self, k, template=None):
        if k > self.vocab_len and self.use_field and template:
            return template[k - self.vocab_len]
        else:
            k = str(k)
            return self.ivocab[k]
            
    def convert_tokens_to_ids(self, sent, template=None):
        return [self.get_word_id(w, template) for w in sent]

    def convert_id_to_tokens(self, word_ids, template_ids=None, remain_eos=False):
        if isinstance(word_ids, list):
            if remain_eos:
                return " ".join([self.get_word(wid, None) for wid in word_ids 
                                 if wid != Constants.PAD])
            else:
                return " ".join([self.get_word(wid, None) for wid in word_ids 
                                 if wid not in [Constants.PAD, Constants.EOS] ])                
        else:
            if remain_eos:
                return " ".join([self.get_word(wid.item(), None) for wid in word_ids 
                                 if wid != Constants.PAD])
            else:
                return " ".join([self.get_word(wid.item(), None) for wid in word_ids 
                                 if wid not in [Constants.PAD, Constants.EOS]])
            
    def convert_template(self, template_ids):
        return [self.get_word(wid) for wid in template_ids if wid != Constants.PAD]

class HDSA_generator():
    
    def __init__(self, archive_file, model_file=None, use_cuda=False):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for HDSA is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(model_dir, 'checkpoints')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)
            
        with open(os.path.join(model_dir, "data/vocab.json"), 'r') as f:
            vocabulary = json.load(f)
        
        vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
        self.tokenizer = Tokenizer(vocab, ivocab, False)
        self.max_seq_length = 50
            
        self.decoder = TableSemanticDecoder(vocab_size=self.tokenizer.vocab_len, d_word_vec=128, n_layers=3, 
                              d_model=128, n_head=4, dropout=0.2)
        self.device = 'cuda' if use_cuda else 'cpu'
        self.decoder.to(self.device)
        checkpoint_file = os.path.join(model_dir, "checkpoints/generator/BERT_dim128_w_domain")
        self.decoder.load_state_dict(torch.load(checkpoint_file))
        
        with open(os.path.join(model_dir, 'data/svdic.pkl'), 'rb') as f:
            self.dic = pickle.load(f)
    
    def init_session(self):
        self.history = []

    def generate(self, usr, pred_hierachical_act_vecs):        
        
        tokens = self.tokenizer.tokenize(usr)
        if self.history:
            tokens = self.history + [Constants.SEP_WORD] + tokens
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[-(self.max_seq_length - 2):]
        tokens = [Constants.CLS_WORD] + tokens + [Constants.SEP_WORD]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        hyps = self.decoder.translate_batch(act_vecs=pred_hierachical_act_vecs, src_seq=input_ids, 
                                       n_bm=2, max_token_seq_len=40)
        pred = self.tokenizer.convert_id_to_tokens(hyps[0])
        
        if not self.history:
            self.history = tokens[1:-1] + [Constants.SEP_WORD] + self.tokenizer.tokenize(pred)
        else:
            self.history = self.history + [Constants.SEP_WORD] + tokens[1:-1] + [Constants.SEP_WORD] + self.tokenizer.tokenize(pred)
        
        return pred

