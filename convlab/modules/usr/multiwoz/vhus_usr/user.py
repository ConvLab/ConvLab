# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from convlab.modules.usr.multiwoz.goal_generator import GoalGenerator
from convlab.modules.usr.multiwoz.vhus_usr.usermodule import VHUS
from convlab.modules.usr.multiwoz.vhus_usr.usermanager import UserDataManager, batch_iter
from convlab.modules.usr.multiwoz.vhus_usr.config import MultiWozConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data):
    if type(data) == dict:
        for k, v in data.items():
            data[k] = v.to(device=DEVICE)
    else:
        for idx, item in enumerate(data):
            data[idx] = item.to(device=DEVICE)
    return data

def padding(old, l):
    """
    pad a list of different lens "old" to the same len "l"
    """
    new = deepcopy(old)
    for i, j in enumerate(new):
        new[i] += [0] * (l - len(j))
        new[i] = j[:l]
    return new

def padding_data(data):
    batch_goals, batch_usrdas, batch_sysdas = deepcopy(data)
    
    batch_input = {}
    posts_length = []
    posts = []
    origin_responses = []
    origin_responses_length = []
    goals_length = []
    goals = []
    terminal = []

    ''' start padding '''
    max_goal_length = max([len(sess_goal) for sess_goal in batch_goals]) # G
    sentence_num = [len(sess) for sess in batch_sysdas]
    # usr begins the session
    max_sentence_num = max(max(sentence_num)-1, 1) # S
        
    # goal & terminal
    for i, l in enumerate(sentence_num):
        goals_length += [len(batch_goals[i])] * l
        goals_padded = batch_goals[i] + [0] * (max_goal_length - len(batch_goals[i]))
        goals += [goals_padded] * l
        terminal += [0] * (l-1) + [1]
        
    # usr
    for sess in batch_usrdas:
        origin_responses_length += [len(sen) for sen in sess]
    max_response_length = max(origin_responses_length) # R
    for sess in batch_usrdas:
        origin_responses += padding(sess, max_response_length)
        
    # sys
    for sess in batch_sysdas:
        sen_length = [len(sen) for sen in sess]
        for j in range(len(sen_length)):
            if j == 0:
                posts_length.append(np.array([1] + [0] * (max_sentence_num - 1)))
            else:
                posts_length.append(np.array(sen_length[:j] + [0] * (max_sentence_num - j)))
    posts_length = np.array(posts_length)
    max_post_length = np.max(posts_length) # P
    for sess in batch_sysdas:
        sen_padded = padding(sess, max_post_length)
        for j, sen in enumerate(sess):
            if j == 0:
                post_single = np.zeros([max_sentence_num, max_post_length], np.int)
            else:
                post_single = posts[-1].copy()
                post_single[j-1, :] = sen_padded[j-1]
            
            posts.append(post_single)
    ''' end padding '''

    batch_input['origin_responses'] = torch.LongTensor(origin_responses) # [B, R]
    batch_input['origin_responses_length'] = torch.LongTensor(origin_responses_length) #[B]
    batch_input['posts_length'] = torch.LongTensor(posts_length) # [B, S]
    batch_input['posts'] = torch.LongTensor(posts) # [B, S, P]
    batch_input['goals_length'] = torch.LongTensor(goals_length) # [B]
    batch_input['goals'] = torch.LongTensor(goals) # [B, G]
    batch_input['terminal'] = torch.Tensor(terminal) # [B]
    
    return batch_input

def kl_gaussian(argu):
    recog_mu, recog_logvar, prior_mu, prior_logvar = argu
    # find the KL divergence between two Gaussian distribution
    loss = 1.0 + (recog_logvar - prior_logvar)
    loss -= (recog_logvar.exp() + torch.pow(recog_mu - prior_mu, 2)) / prior_logvar.exp()
    kl_loss = -0.5 * loss.sum(dim=1)
    avg_kl_loss = kl_loss.mean()
    return avg_kl_loss

class UserNeural():
    def __init__(self, pretrain=False):
    
        config = MultiWozConfig()
        manager = UserDataManager(config.data_dir, config.data_file)
        voc_goal_size, voc_usr_size, voc_sys_size = manager.get_voc_size()
        self.user = VHUS(config, voc_goal_size, voc_usr_size, voc_sys_size).to(device=DEVICE)
        self.optim = optim.Adam(self.user.parameters(), lr=config.lr_simu)
        self.goal_gen = GoalGenerator(config.data_dir+'/goal/goal_model.pkl')
        self.cfg = config
        self.manager = manager
        self.user.eval()
        
        if pretrain:
            self.print_per_batch = config.print_per_batch
            self.save_dir = config.save_dir
            self.save_per_epoch = config.save_per_epoch
            seq_goals, seq_usr_dass, seq_sys_dass = manager.data_loader_seg()
            train_goals, train_usrdas, train_sysdas, \
            test_goals, test_usrdas, test_sysdas, \
            val_goals, val_usrdas, val_sysdas = manager.train_test_val_split_seg(
                seq_goals, seq_usr_dass, seq_sys_dass)
            self.data_train = (train_goals, train_usrdas, train_sysdas, config.batchsz)
            self.data_valid = (val_goals, val_usrdas, val_sysdas, config.batchsz)
            self.data_test = (test_goals, test_usrdas, test_sysdas, config.batchsz)
            self.nll_loss = nn.NLLLoss(ignore_index=0) # PAD=0
            self.bce_loss = nn.BCEWithLogitsLoss()
        else:
            self.load(config.load)
            
    def user_loop(self, data):
        batch_input = to_device(padding_data(data))
        a_weights, t_weights, argu = self.user(batch_input['goals'], batch_input['goals_length'], \
                                         batch_input['posts'], batch_input['posts_length'], batch_input['origin_responses'])
        
        loss_a, targets_a = 0, batch_input['origin_responses'][:, 1:] # remove sos_id
        for i, a_weight in enumerate(a_weights):
            loss_a += self.nll_loss(a_weight, targets_a[:, i])
        loss_a /= i
        loss_t = self.bce_loss(t_weights, batch_input['terminal'])
        loss_a += self.cfg.alpha * kl_gaussian(argu)
        return loss_a, loss_t
        
    def imitating(self, epoch):
        """
        train the user simulator by simple imitation learning (behavioral cloning)
        """
        self.user.train()
        a_loss, t_loss = 0., 0.
        data_train_iter = batch_iter(self.data_train[0], self.data_train[1], self.data_train[2], self.data_train[3])
        for i, data in enumerate(data_train_iter):
            self.optim.zero_grad()
            loss_a, loss_t = self.user_loop(data)
            a_loss += loss_a.item()
            t_loss += loss_t.item()
            loss = loss_a + loss_t
            loss.backward()
            self.optim.step()
            
            if (i+1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                t_loss /= self.print_per_batch
                logging.debug('<<user simulator>> epoch {}, iter {}, loss_a:{}, loss_t:{}'.format(epoch, i, a_loss, t_loss))
                a_loss, t_loss = 0., 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.user.eval()
        
    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the user simulator fit on the training dataset
        """        
        a_loss, t_loss = 0., 0.
        data_valid_iter = batch_iter(self.data_valid[0], self.data_valid[1], self.data_valid[2], self.data_valid[3])
        for i, data in enumerate(data_valid_iter):
            loss_a, loss_t = self.user_loop(data)
            a_loss += loss_a.item()
            t_loss += loss_t.item()
            
        a_loss /= i
        t_loss /= i
        logging.debug('<<user simulator>> validation, epoch {}, loss_a:{}, loss_t:{}'.format(epoch, a_loss, t_loss))
        loss = a_loss + t_loss
        if loss < best:
            logging.info('<<user simulator>> best model saved')
            best = loss
            self.save(self.save_dir, 'best')
            
        a_loss, t_loss = 0., 0.
        data_test_iter = batch_iter(self.data_test[0], self.data_test[1], self.data_test[2], self.data_test[3])
        for i, data in enumerate(data_test_iter):
            loss_a, loss_t = self.user_loop(data)
            a_loss += loss_a.item()
            t_loss += loss_t.item()
            
        a_loss /= i
        t_loss /= i
        logging.debug('<<user simulator>> test, epoch {}, loss_a:{}, loss_t:{}'.format(epoch, a_loss, t_loss))
        return best
        
    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.user.state_dict(), directory + '/' + str(epoch) + '_simulator.mdl')
        logging.info('<<user simulator>> epoch {}: saved network to mdl'.format(epoch))
    
    def load(self, filename):
        user_mdl = filename + '_simulator.mdl'
        if os.path.exists(user_mdl):
            self.user.load_state_dict(torch.load(user_mdl))
            logging.info('<<user simulator>> loaded checkpoint from file: {}'.format(user_mdl))
    
    def init_session(self):
        self.time_step = -1
        self.topic = 'NONE'
        self.goal = self.goal_gen.get_user_goal()
        self.goal_input = torch.LongTensor(self.manager.get_goal_id(self.manager.usrgoal2seq(self.goal)))
        self.goal_len_input = torch.LongTensor([len(self.goal_input)]).squeeze()
        self.sys_da_id_stack = [] # to save sys da history

    def predict(self, state, sys_action):
        """
        Predict an user act based on state and preorder system action.
        Args:
            state (tuple): Dialog state.
            sys_action (tuple): Preorder system action.s
        Returns:
            usr_action (tuple): User act.
            session_over (boolean): True to terminate session, otherwise session continues.
            reward (float): Reward given by user.
        """
        sys_seq_turn = self.manager.sysda2seq(self.manager.ref_data2stand(sys_action), self.goal)
        self.sys_da_id_stack += self.manager.get_sysda_id([sys_seq_turn])
        sys_seq_len = torch.LongTensor([max(len(sen), 1) for sen in self.sys_da_id_stack])
        max_sen_len = sys_seq_len.max().item()
        sys_seq = torch.LongTensor(padding(self.sys_da_id_stack, max_sen_len))
        usr_a, terminal = self.user.select_action(self.goal_input, self.goal_len_input, sys_seq, sys_seq_len)
        usr_action = self._dict_to_vec(self.manager.usrseq2da(self.manager.id2sentence(usr_a), self.goal))
        
        return usr_action, terminal, None

if __name__ == '__main__':
    manager = UserDataManager('../../../../data/multiwoz', 'annotated_user_da_with_span_full.json')
    seq_goals, seq_usr_dass, seq_sys_dass = manager.data_loader_seg()
    train_goals, train_usrdas, train_sysdas, \
    test_goals, test_usrdas, test_sysdas, \
    val_goals, val_usrdas, val_sysdas = manager.train_test_val_split_seg(
        seq_goals, seq_usr_dass, seq_sys_dass)
    data_train = batch_iter(train_goals, train_usrdas, train_sysdas, 32)
    data_valid = batch_iter(val_goals, val_usrdas, val_sysdas, 32)
    data_test = batch_iter(test_goals, test_usrdas, test_sysdas, 32)
    for data in data_train:
        batch_input = to_device(padding_data(data))
        break
    for k, v in batch_input.items():
        print(k, v.shape)
    voc_goal_size, voc_usr_size, voc_sys_size = manager.get_voc_size()
    cfg = MultiWozConfig()
    user = VHUS(cfg, voc_goal_size, voc_usr_size, voc_sys_size)
    
    a_weights, t_weights, _ = user(batch_input['goals'], batch_input['goals_length'], batch_input['posts'], batch_input['posts_length'])#, batch_input['origin_responses'])
    print(len(a_weights)) #[L, B, V]
    print(t_weights.shape) #[B]
