# -*- coding: utf-8 -*-
import os
from convlab.modules.policy.user.policy import UserPolicy
from convlab.modules.usr.multiwoz.vhus_usr.user import UserNeural

DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "nlg-sclstm-multiwoz.zip")

class UserPolicyVHUS(UserPolicy):

    def __init__(self):
        self.user = UserNeural()
        
    def init_session(self):
        self.user.init_session()
        
    def predict(self, state, sys_action):
        usr_action, terminal, reward = self.user.predict(state, sys_action)
        return usr_action, terminal, reward