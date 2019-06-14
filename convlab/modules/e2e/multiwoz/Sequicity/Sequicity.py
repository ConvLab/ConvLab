# -*- coding: utf-8 -*-

# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import zipfile

import numpy as np
import torch
from nltk import word_tokenize
from torch.autograd import Variable

from convlab.lib.file_util import cached_path
from convlab.modules.e2e.multiwoz.Sequicity.config import global_config as cfg
from convlab.modules.e2e.multiwoz.Sequicity.model import Model
from convlab.modules.e2e.multiwoz.Sequicity.reader import pad_sequences
from convlab.modules.e2e.multiwoz.Sequicity.tsd_net import cuda_
from convlab.modules.policy.system.policy import SysPolicy

DEFAULT_CUDA_DEVICE=-1
DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "Sequicity.rar")

def denormalize(uttr):
    uttr = uttr.replace(' -s', 's')
    uttr = uttr.replace(' -ly', 'ly')
    uttr = uttr.replace(' -er', 'er')
    return uttr

class Sequicity(SysPolicy):
    def __init__(self, 
                 archive_file=DEFAULT_ARCHIVE_FILE, 
                 model_file=None):
        SysPolicy.__init__(self)
        
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for Sequicity is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(model_dir, 'data')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)
        
        cfg.init_handler('tsdf-multiwoz')
        
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.m = Model('multiwoz')
        self.m.count_params()
        self.m.load_model()
        self.reset()
        
    def reset(self):
        self.kw_ret = dict({'func':self.z2degree})
    
    def z2degree(self, gen_z):
        gen_bspan = self.m.reader.vocab.sentence_decode(gen_z, eos='EOS_Z2')
        constraint_request = gen_bspan.split()
        constraints = constraint_request[:constraint_request.index('EOS_Z1')] if 'EOS_Z1' \
            in constraint_request else constraint_request
        for j, ent in enumerate(constraints):
            constraints[j] = ent.replace('_', ' ')
        degree = self.m.reader.db_search(constraints)
        degree_input_list = self.m.reader._degree_vec_mapping(len(degree))
        degree_input = cuda_(Variable(torch.Tensor(degree_input_list).unsqueeze(0)))
        return degree, degree_input
    
    def predict(self, usr):            
        print('usr:', usr)
        usr = word_tokenize(usr.lower())
        usr_words = usr + ['EOS_U']
        u_len = np.array([len(usr_words)])
        usr_indices = self.m.reader.vocab.sentence_encode(usr_words)
        u_input_np = np.array(usr_indices)[:, np.newaxis]
        u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
        m_idx, z_idx, degree = self.m.m(mode='test', degree_input=None, z_input=None,
                                        u_input=u_input, u_input_np=u_input_np, u_len=u_len,
                                        m_input=None, m_input_np=None, m_len=None,
                                        turn_states=None, **self.kw_ret)
        venue = random.sample(degree, 1)[0] if degree else dict()
        l = [self.m.reader.vocab.decode(_) for _ in m_idx[0]]
        if 'EOS_M' in l:
            l = l[:l.index('EOS_M')]
        l_origin = []
        for word in l:
            if 'SLOT' in word:
                word = word[:-5]
                if word in venue.keys():
                    value = venue[word]
                    if value != '?':
                        l_origin.append(value)
            else:
                l_origin.append(word)
        sys = ' '.join(l_origin)
        sys = denormalize(sys)
        print('sys:', sys)
        if cfg.prev_z_method == 'separate':
            eob = self.m.reader.vocab.encode('EOS_Z2')
            if eob in z_idx[0] and z_idx[0].index(eob) != len(z_idx[0]) - 1:
                idx = z_idx[0].index(eob)
                z_idx[0] = z_idx[0][:idx + 1]
            for j, word in enumerate(z_idx[0]):
                if word >= cfg.vocab_size:
                    z_idx[0][j] = 2 #unk
            prev_z_input_np = pad_sequences(z_idx, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_len = np.array([len(_) for _ in z_idx])
            prev_z_input = cuda_(Variable(torch.from_numpy(prev_z_input_np).long()))
            self.kw_ret['prev_z_len'] = prev_z_len
            self.kw_ret['prev_z_input'] = prev_z_input
            self.kw_ret['prev_z_input_np'] = prev_z_input_np
        return sys