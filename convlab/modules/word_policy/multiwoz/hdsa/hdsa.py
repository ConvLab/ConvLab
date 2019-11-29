# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:58:00 2019

@author: truthless
"""
import os
from convlab.modules.word_policy.multiwoz.hdsa.predictor import HDSA_predictor
from convlab.modules.word_policy.multiwoz.hdsa.generator import HDSA_generator
from convlab.modules.policy.system.policy import SysPolicy

DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "hdsa.zip")

class HDSA(SysPolicy):
    
    def __init__(self, archive_file=DEFAULT_ARCHIVE_FILE, model_file=None, use_cuda=False):
        self.predictor = HDSA_predictor(archive_file, model_file, use_cuda)
        self.generator = HDSA_generator(archive_file, model_file, use_cuda)
        
    def init_session(self):
        self.generator.init_session()
        
    def predict(self, state):
        
        act, kb = self.predictor.predict(state)
        response = self.generator.generate(state, act, kb)
        
        return response
