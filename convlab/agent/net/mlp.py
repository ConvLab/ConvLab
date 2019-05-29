# Modified by Microsoft Corporation.
# Licensed under the MIT license.

from convlab.agent.net import net_util
from convlab.agent.net.base import Net
from convlab.lib import logger, math_util, util
import numpy as np
import pydash as ps
import torch
import torch.nn as nn

logger = logger.get_logger(__name__)


class MLPNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network
    If more than 1 output tensors, will create a self.model_tails instead of making last layer part of self.model

    e.g. net_spec
    "net": {
        "type": "MLPNet",
        "shared": true,
        "hid_layers": [32],
        "hid_layers_activation": "relu",
        "init_fn": "xavier_uniform_",
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "MSELoss"
        },
        "optim_spec": {
          "name": "Adam",
          "lr": 0.02
        },
        "lr_scheduler_spec": {
            "name": "StepLR",
            "step_size": 30,
            "gamma": 0.1
        },
        "update_type": "replace",
        "update_frequency": 1,
        "polyak_coef": 0.9,
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        net_spec:
        hid_layers: list containing dimensions of the hidden layers
        hid_layers_activation: activation function for the hidden layers
        init_fn: weight initialization function
        clip_grad_val: clip gradient norm if value is not None
        loss_spec: measure of error between model predictions and correct outputs
        optim_spec: parameters for initializing the optimizer
        lr_scheduler_spec: Pytorch optim.lr_scheduler
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_coef: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
        nn.Module.__init__(self)
        super(MLPNet, self).__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            init_fn=None,
            clip_grad_val=None,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_scheduler_spec=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'shared',
            'hid_layers',
            'hid_layers_activation',
            'init_fn',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])

        dims = [self.in_dim] + self.hid_layers
        self.model = net_util.build_sequential(dims, self.hid_layers_activation)
        # add last layer with no activation
        # tails. avoid list for single-tail for compute speed
        if ps.is_integer(self.out_dim):
            self.model_tail = nn.Linear(dims[-1], self.out_dim)
        else:
            self.model_tails = nn.ModuleList([nn.Linear(dims[-1], out_d) for out_d in self.out_dim])

        net_util.init_layers(self, self.init_fn)
        for module in self.modules():
            module.to(self.device)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self, self.lr_scheduler_spec)

    def __str__(self):
        return super(MLPNet, self).__str__() + f'\noptim: {self.optim}'

    def forward(self, x):
        '''The feedforward step'''
        x = self.model(x)
        if hasattr(self, 'model_tails'):
            outs = []
            for model_tail in self.model_tails:
                outs.append(model_tail(x))
            return outs
        else:
            return self.model_tail(x)

    def training_step(self, x=None, y=None, loss=None, retain_graph=False, lr_clock=None):
        '''
        Takes a single training step: one forward and one backwards pass
        More most RL usage, we have custom, often complicated, loss functions. Compute its value and put it in a pytorch tensor then pass it in as loss
        '''
        if hasattr(self, 'model_tails') and x is not None:
            raise ValueError('Loss computation from x,y not supported for multitails')
        self.lr_scheduler.step(epoch=ps.get(lr_clock, 'total_t'))
        self.train()
        self.optim.zero_grad()
        if loss is None:
            out = self(x)
            loss = self.loss_fn(out, y)
        assert not torch.isnan(loss).any(), loss
        if net_util.to_assert_trained():
            assert_trained = net_util.gen_assert_trained(self)
        loss.backward(retain_graph=retain_graph)
        if self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        self.optim.step()
        if net_util.to_assert_trained():
            assert_trained(self, loss)
            self.store_grad_norms()
        logger.debug(f'Net training_step loss: {loss}')
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model
        returns: network output given input x
        '''
        self.eval()
        return self(x)


class HydraMLPNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network with multiple state and action heads, and a single shared body.

    e.g. net_spec
    "net": {
        "type": "HydraMLPNet",
        "shared": true,
        "hid_layers": [
            [[32],[32]], # 2 heads with hidden layers
            [64], # body
            [] # tail, no hidden layers
        ],
        "hid_layers_activation": "relu",
        "init_fn": "xavier_uniform_",
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "MSELoss"
        },
        "optim_spec": {
          "name": "Adam",
          "lr": 0.02
        },
        "lr_scheduler_spec": {
            "name": "StepLR",
            "step_size": 30,
            "gamma": 0.1
        },
        "update_type": "replace",
        "update_frequency": 1,
        "polyak_coef": 0.9,
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        Multi state processing heads, single shared body, and multi action tails.
        There is one state and action head per body/environment
        Example:

          env 1 state       env 2 state
         _______|______    _______|______
        |    head 1    |  |    head 2    |
        |______________|  |______________|
                |                  |
                |__________________|
         ________________|_______________
        |          Shared body           |
        |________________________________|
                         |
                 ________|_______
                |                |
         _______|______    ______|_______
        |    tail 1    |  |    tail 2    |
        |______________|  |______________|
                |                |
           env 1 action      env 2 action
        '''
        nn.Module.__init__(self)
        super(HydraMLPNet, self).__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            init_fn=None,
            clip_grad_val=None,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_scheduler_spec=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'hid_layers',
            'hid_layers_activation',
            'init_fn',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])
        assert len(self.hid_layers) == 3, 'Your hidden layers must specify [*heads], [body], [*tails]. If not, use MLPNet'
        assert isinstance(self.in_dim, list), 'Hydra network needs in_dim as list'
        assert isinstance(self.out_dim, list), 'Hydra network needs out_dim as list'
        self.head_hid_layers = self.hid_layers[0]
        self.body_hid_layers = self.hid_layers[1]
        self.tail_hid_layers = self.hid_layers[2]
        if len(self.head_hid_layers) == 1:
            self.head_hid_layers = self.head_hid_layers * len(self.in_dim)
        if len(self.tail_hid_layers) == 1:
            self.tail_hid_layers = self.tail_hid_layers * len(self.out_dim)

        self.model_heads = self.build_model_heads(in_dim)
        heads_out_dim = np.sum([head_hid_layers[-1] for head_hid_layers in self.head_hid_layers])
        dims = [heads_out_dim] + self.body_hid_layers
        self.model_body = net_util.build_sequential(dims, self.hid_layers_activation)
        self.model_tails = self.build_model_tails(out_dim)

        net_util.init_layers(self, self.init_fn)
        for module in self.modules():
            module.to(self.device)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self, self.lr_scheduler_spec)

    def __str__(self):
        return super(HydraMLPNet, self).__str__() + f'\noptim: {self.optim}'

    def build_model_heads(self, in_dim):
        '''Build each model_head. These are stored as Sequential models in model_heads'''
        assert len(self.head_hid_layers) == len(in_dim), 'Hydra head hid_params inconsistent with number in dims'
        model_heads = nn.ModuleList()
        for in_d, hid_layers in zip(in_dim, self.head_hid_layers):
            dims = [in_d] + hid_layers
            model_head = net_util.build_sequential(dims, self.hid_layers_activation)
            model_heads.append(model_head)
        return model_heads

    def build_model_tails(self, out_dim):
        '''Build each model_tail. These are stored as Sequential models in model_tails'''
        model_tails = nn.ModuleList()
        if ps.is_empty(self.tail_hid_layers):
            for out_d in out_dim:
                model_tails.append(nn.Linear(self.body_hid_layers[-1], out_d))
        else:
            assert len(self.tail_hid_layers) == len(out_dim), 'Hydra tail hid_params inconsistent with number out dims'
            for out_d, hid_layers in zip(out_dim, self.tail_hid_layers):
                dims = hid_layers
                model_tail = net_util.build_sequential(dims, self.hid_layers_activation)
                model_tail.add_module(str(len(model_tail)), nn.Linear(dims[-1], out_d))
                model_tails.append(model_tail)
        return model_tails

    def forward(self, xs):
        '''The feedforward step'''
        head_xs = []
        for model_head, x in zip(self.model_heads, xs):
            head_xs.append(model_head(x))
        head_xs = torch.cat(head_xs, dim=-1)
        body_x = self.model_body(head_xs)
        outs = []
        for model_tail in self.model_tails:
            outs.append(model_tail(body_x))
        return outs

    def training_step(self, xs=None, ys=None, loss=None, retain_graph=False, lr_clock=None):
        '''
        Takes a single training step: one forward and one backwards pass. Both x and y are lists of the same length, one x and y per environment
        '''
        self.lr_scheduler.step(epoch=ps.get(lr_clock, 'total_t'))
        self.train()
        self.optim.zero_grad()
        if loss is None:
            outs = self(xs)
            total_loss = torch.tensor(0.0, device=self.device)
            for out, y in zip(outs, ys):
                loss = self.loss_fn(out, y)
                total_loss += loss
            loss = total_loss
        assert not torch.isnan(loss).any(), loss
        if net_util.to_assert_trained():
            assert_trained = net_util.gen_assert_trained(self)
        loss.backward(retain_graph=retain_graph)
        if self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        self.optim.step()
        if net_util.to_assert_trained():
            assert_trained(self, loss)
            self.store_grad_norms()
        logger.debug(f'Net training_step loss: {loss}')
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model
        returns: network output given input x
        '''
        self.eval()
        return self(x)


class DuelingMLPNet(MLPNet):
    '''
    Class for generating arbitrary sized feedforward neural network, with dueling heads. Intended for Q-Learning algorithms only.
    Implementation based on "Dueling Network Architectures for Deep Reinforcement Learning" http://proceedings.mlr.press/v48/wangf16.pdf

    e.g. net_spec
    "net": {
        "type": "DuelingMLPNet",
        "shared": true,
        "hid_layers": [32],
        "hid_layers_activation": "relu",
        "init_fn": "xavier_uniform_",
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "MSELoss"
        },
        "optim_spec": {
          "name": "Adam",
          "lr": 0.02
        },
        "lr_scheduler_spec": {
            "name": "StepLR",
            "step_size": 30,
            "gamma": 0.1
        },
        "update_type": "replace",
        "update_frequency": 1,
        "polyak_coef": 0.9,
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            init_fn=None,
            clip_grad_val=None,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_scheduler_spec=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'shared',
            'hid_layers',
            'hid_layers_activation',
            'init_fn',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])

        # Guard against inappropriate algorithms and environments
        # Build model body
        dims = [self.in_dim] + self.hid_layers
        self.model_body = net_util.build_sequential(dims, self.hid_layers_activation)
        # output layers
        self.v = nn.Linear(dims[-1], 1)  # state value
        self.adv = nn.Linear(dims[-1], out_dim)  # action dependent raw advantage
        net_util.init_layers(self, self.init_fn)
        for module in self.modules():
            module.to(self.device)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self, self.lr_scheduler_spec)

    def forward(self, x):
        '''The feedforward step'''
        x = self.model_body(x)
        state_value = self.v(x)
        raw_advantages = self.adv(x)
        out = math_util.calc_q_value_logits(state_value, raw_advantages)
        return out
