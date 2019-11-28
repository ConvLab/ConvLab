# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:58:00 2019

@author: truthless
"""
import os
from convlab.modules.word_policy.multiwoz.hdsa.predictor import HDSA_predictor
from convlab.modules.word_policy.multiwoz.hdsa.generator import HDSA_generator
from convlab.modules.policy.system.policy import SysPolicy
from convlab.modules.util.multiwoz.dialog_manager import Dialog_manager

DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "hdsa.zip")

class HDSA(SysPolicy):
    
    def __init__(self, archive_file=DEFAULT_ARCHIVE_FILE, model_file=None, use_cuda=False):
        self.predictor = HDSA_predictor(archive_file, model_file, use_cuda)
        self.generator = HDSA_generator(archive_file, model_file, use_cuda)
        self.manager = Dialog_manager()
        
    def init_session(self):
        self.generator.init_session()
        
    def predict(self, state):
        
        parsed_state, domain = self.manager.parse(state)
        usr = state['history'][-1][-1]
        sys = state['history'][-1][-2] if len(state['history'][-1]) > 1 else ''
        sentence, domain = self.manager.process(usr, parsed_state, domain)
        db_num, db_result = self.manager.update(parsed_state, domain)
        
        act = self.predictor.predict(usr, sys, db_result, domain)
        response = self.generator.generate(sentence, act)
        
        response = self.manager.relexilize(response, domain)
        
        return response
