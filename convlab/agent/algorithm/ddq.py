# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch

from convlab.agent import memory
from convlab.agent import net
from convlab.agent.algorithm.dqn import DQN
from convlab.agent.net import net_util
from convlab.lib import logger, util
from convlab.lib.decorator import lab_api

logger = logger.get_logger(__name__)


class DDQ(DQN):
    '''
    Implementation of a simple DDQ algorithm.
    '''

    @lab_api
    def init_algorithm_params(self):
        # set default
        util.set_attr(self, dict(
            action_pdtype='Argmax',
            action_policy='epsilon_greedy',
            explore_var_spec=None,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # explore_var is epsilon, tau or etc. depending on the action policy
            # these control the trade off between exploration and exploitaton
            'explore_var_spec',
            'gamma',  # the discount factor
            'training_batch_iter',  # how many gradient updates per batch
            'training_iter',  # how many batches to train each time
            'training_frequency',  # how often to train (once a few timesteps)
            'training_start_step',  # how long before starting training
            'planning_steps', # how many more step for planning
        ])
        super().init_algorithm_params()

    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize the neural network used to learn the Q function from the spec'''
        if self.algorithm_spec['name'] == 'VanillaDQN':
            assert all(k not in self.net_spec for k in ['update_type', 'update_frequency', 'polyak_coef']), 'Network update not available for VanillaDQN; use DQN.'
        in_dim = self.body.state_dim
        out_dim = net_util.get_out_dim(self.body)
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim)
        self.net_names = ['net']
        # init net optimizer and its lr scheduler
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec)
        net_util.set_global_nets(self, global_nets)
        self.post_init_nets()


        # load a pre-trained world model
        # TODO: 
        # 1. This pre-trained model is pretty good at fisrt. Should change to initialize randomly
        # 2. This model restricted to MultiWoz data. Should expand to any type
        self.world_model = WorldModel(self.body)

    @lab_api
    def train(self):
        '''
        Completes one training step for the agent if it is time to train.
        i.e. the environment timestep is greater than the minimum training timestep and a multiple of the training_frequency.
        Each training step consists of sampling n batches from the agent's memory.
        For each of the batches, the target Q values (q_targets) are computed and a single training step is taken k times
        Otherwise this function does nothing.
        '''
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock
        if self.to_train == 1:
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_iter):
                batch = self.sample()
                clock.set_batch_size(len(batch))
                for _ in range(self.training_batch_iter):
                    loss = self.calc_q_loss(batch)
                    self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
                    total_loss += loss

                    # [model-based]: train world_model on real data
                    self.world_model.train(batch)

            # [model-based]: plan more steps with world_model
            for _ in range(self.planning_steps):
                for _ in range(self.training_iter):
                    batch = self.sample()
                    clock.set_batch_size(len(batch))
                    for _ in range(self.training_batch_iter):
                        fake_batch = self.world_model.create_batch(batch)
                        loss = self.calc_q_loss(fake_batch) # this also inluences the priority in memory
                        self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
    
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class WorldModel(nn.Module):

    def __init__(self, body, pretrain=False):
        self.body = body
        self.init_nets()
        
        # load pretrain models

    def init_nets(self):
        '''Initialize the neural network used to learn the Q function from the spec'''

        # build world_model nets
        # its a multi-task model
        state_dim = self.body.state_dim
        action_dim = self.body.action_dim

        hidden_dim = 12
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.hidden_layer = nn.Linear(2*hidden_dim, hidden_dim)
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.terminal_head = nn.Linear(state_dim, 1)

        # init net optimizer and its lr scheduler
        self.optim = optim.Adam(lr=0.01)

    def forward(self, s, a):
        se = self.state_encoder(s)
        ae = self.action_encoder(a)
        hid = self.hidden_layer(torch.cat([se,ae]))
        r = self.reward_head(hid)
        ns = self.state_head(hid)
        d =  self.terminal_head(hid)
        return ns, r, d


    def predict(self, states, actions)
        """
        Predict an user act based on state and preorder system action.
        Args:
        Returns:
        """
        ns, r, t = self.forward(states, actions)
        return ns, r, F.sigmoid(t)

    def train(self, batch):
        self.optim.zero_grad()
        states = batch["states"]
        actions = batch["actions"]
        next_states, rewards, dones = self.predict(states, actions)

        # compute loss
        loss_s = F.mse_loss(batch["next_states"], next_states) # TODO: state loss
        loss_r = F.mse_loss(batch["rewards"], rewards)
        loss_t = F.binary_cross_entropy_with_logits(batch["dones"], dones)
        loss = loss_s + loss_r + loss_t

        loss.backward()
        self.optim.step()

    def create_batch(self, batch):
        states = batch["states"]
        actions = batch["actions"]
        next_states, rewards, dones = self.predict(states, actions)
        new_batch = {}
        new_batch["states"] = states
        new_batch["next_states"] = next_states
        new_batch["actions"] = actions 
        new_batch["rewards"] = rewards
        new_batch["dones"] = dones
        return new_batch

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

