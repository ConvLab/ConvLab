# Modified by Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
import os
class MultiWozConfig():
    
    def __init__(self):
        self.data_file = 'annotated_user_da_with_span_full.json'
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))),
                                     'data/multiwoz')
        
        self.print_per_batch = 400
        self.batchsz = 32
        self.save_per_epoch = 2
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        self.load = os.path.join(self.save_dir, 'best')
        
        self.lr_simu = 1e-3
        self.hu_dim = 200 # for user module
        self.eu_dim = 150
        self.max_ulen = 20
        self.alpha = 0.01
