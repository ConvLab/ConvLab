# Modified by Microsoft Corporation.
# Licensed under the MIT license.

'''
The random agent algorithm
For basic dev purpose.
'''
from copy import deepcopy

import pydash as ps

from convlab.agent.algorithm import policy_util
from convlab.agent.algorithm.base import Algorithm
from convlab.lib import logger, util
from convlab.lib.decorator import lab_api
from convlab.modules import policy, word_policy, e2e

logger = logger.get_logger(__name__)


class ExternalPolicy(Algorithm):
    '''
    Example Random agent that works in both discrete and continuous envs
    '''

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
        ))
        util.set_attr(self, self.algorithm_spec, [
            'policy_name',
            'action_pdtype',
            'action_policy',
        ])
        self.action_policy = getattr(policy_util, self.action_policy)
        self.policy = None 
        if 'word_policy' in self.algorithm_spec:
            params = deepcopy(ps.get(self.algorithm_spec, 'word_policy'))
            PolicyClass = getattr(word_policy, params.pop('name'))
        elif 'e2e' in self.algorithm_spec:
            params = deepcopy(ps.get(self.algorithm_spec, 'e2e'))
            PolicyClass = getattr(e2e, params.pop('name'))
        else:
            params = deepcopy(ps.get(self.algorithm_spec, 'policy'))
            PolicyClass = getattr(policy, params.pop('name'))
        self.policy = PolicyClass(**params) 

    def reset(self):
        self.policy.init_session()

    @lab_api
    def init_nets(self, global_nets=None):
        pass

    @lab_api
    def act(self, state):
        action = self.policy.predict(state)
        return action

    @lab_api
    def sample(self):
        pass

    @lab_api
    def train(self):
        pass

    @lab_api
    def update(self):
        pass