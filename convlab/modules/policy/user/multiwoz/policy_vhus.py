# -*- coding: utf-8 -*-
import os
from convlab.modules.policy.user.policy import UserPolicy
from convlab.modules.usr.multiwoz.vhus_usr.user import UserNeural

class UserPolicyVHUS(UserPolicy):

    def __init__(self):
        self.user = UserNeural()
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 
                            'usr/multiwoz/vhus_user/model/best')
        self.user.load(path)
        
    def init_session(self):
        self.user.init_session()
        self.domain_goals = self.user.goal
        
    def predict(self, state, sys_action):
        usr_action, terminal = self.user.predict(state, sys_action)
        return usr_action, terminal, 0