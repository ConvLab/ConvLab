# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torch.nn.functional as F

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

        # [model-based] init world_model_nets
        self.state_dim = self.body.state_dim
        self.action_dim = net_util.get_out_dim(self.body)
        in_dim = [self.state_dim, self.action_dim]
        out_dim = [self.state_dim, 1, 1]
        
        WorldNetClass = getattr(net, self.world_net_spec['type'])
        self.world_net = WorldNetClass(self.world_net_spec,in_dim, out_dim)
        self.world_net_names = ['world_net']
        self.world_optim = net_util.get_optim(self.world_net, self.world_net.optim_spec)
        self.world_lr_scheduler = net_util.get_lr_scheduler(self.world_optim, self.world_net.lr_scheduler_spec)
         
        print(self.world_net)

        # initialize policy net 
        super().init_nets(global_nets)

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
                    self.train_world_model(batch)

            # [model-based]: plan more steps with world_model
            for _ in range(self.planning_steps):
                for _ in range(self.training_iter):
                    batch = self.sample()
                    clock.set_batch_size(len(batch))
                    for _ in range(self.training_batch_iter):
                        fake_batch = self.planning(batch)
                        loss = self.calc_q_loss(fake_batch) # this also inluences the priority in memory
                        self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
    
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan


    def train_world_model(self, batch):
        # zero_grad
        self.world_optim.zero_grad()

        # get predictions
        states_raw  = batch["states"]
        actions_raw = batch["actions"]
        states = states_raw
        actions = F.one_hot(actions_raw.long(), self.action_dim).float()
        next_states, rewards, dones = self.world_net([states, actions])
        rewards = rewards.view(-1)
        dones = dones.view(-1)

        # compute loss
        loss_func_state = torch.nn.BCEWithLogitsLoss()
        loss_s = loss_func_state(next_states, batch["next_states"])
        loss_func_reward = torch.nn.MSELoss()
        loss_r = loss_func_reward(rewards, batch["rewards"])
        loss_func_terminal = torch.nn.BCEWithLogitsLoss()
        loss_t = loss_func_terminal(dones, batch["dones"])
        loss = loss_s + loss_r + loss_t

        # update
        loss.backward()
        self.world_optim.step()

    def planning(self, batch):
        # get predictions
        states_raw  = batch["states"]
        actions_raw = batch["actions"]
        states = states_raw
        actions = F.one_hot(actions_raw.long(), self.action_dim).float()
        next_states, rewards, dones = self.world_net([states, actions])
        rewards = rewards.view(-1)
        dones = dones.view(-1)

        # sample next_states/dones to [0,1]
        m = torch.distributions.Bernoulli(torch.sigmoid(next_states))
        next_states = m.sample()
        m = torch.distributions.Bernoulli(torch.sigmoid(dones))
        dones = m.sample()

        # create new batch
        new_batch = {}
        new_batch["states"] = states_raw
        new_batch["next_states"] = next_states
        new_batch["actions"] = actions_raw
        new_batch["rewards"] = rewards
        new_batch["dones"] = dones
        return new_batch
