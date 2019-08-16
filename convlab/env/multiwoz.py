# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
from copy import deepcopy

import numpy as np
import pydash as ps
from gym import spaces

from convlab import evaluator
import convlab.modules.nlg.multiwoz as nlg
import convlab.modules.nlu.multiwoz as nlu
import convlab.modules.policy.system.multiwoz as sys_policy
import convlab.modules.policy.user.multiwoz as user_policy
from convlab.modules.policy.user.multiwoz import UserPolicyAgendaMultiWoz
from convlab.modules.usr import UserSimulator
from convlab.env.base import BaseEnv, set_gym_space_attr
from convlab.lib import logger, util
from convlab.lib.decorator import lab_api
from convlab.modules.action_decoder.multiwoz.multiwoz_vocab_action_decoder import ActionVocab
from convlab.modules.policy.system.multiwoz.rule_based_multiwoz_bot import RuleBasedMultiwozBot

logger = logger.get_logger(__name__)

class State(object):
    def __init__(self, state=None, reward=None, done=None):
        self.states = [state]
        self.rewards = [reward]
        self.local_done = [done]


class MultiWozEnvironment(object):
    def __init__(self, env_spec, worker_id=None, action_dim=300):
        self.env_spec = env_spec
        self.worker_id = worker_id
        self.observation_space = None 
        self.action_space = None

        self.agenda = UserPolicyAgendaMultiWoz()  # Agenda-based Simulator (act-in act-out)
        if 'user_policy' in self.env_spec:
            params = deepcopy(ps.get(self.env_spec, 'user_policy'))
            AgendaClass = getattr(user_policy, params.pop('name'))
            self.agenda = AgendaClass()

        self.nlu = None
        if 'nlu' in self.env_spec:
            params = deepcopy(ps.get(self.env_spec, 'nlu'))
            if not params['name']:
                self.nlu = None
            else:
                NluClass = getattr(nlu, params.pop('name'))
                self.nlu = NluClass(**params)

        self.nlg = None
        if 'nlg' in self.env_spec:
            params = deepcopy(ps.get(self.env_spec, 'nlg'))
            if not params['name']:
                self.nlg = None
            else:
                NlgClass = getattr(nlg, params.pop('name'))
                self.nlg = NlgClass(**params)

        self.sys_policy = RuleBasedMultiwozBot()
        if 'sys_policy' in self.env_spec:
            params = deepcopy(ps.get(self.env_spec, 'sys_policy'))
            SysPolicy = getattr(sys_policy, params.pop('name'))
            self.sys_policy = SysPolicy()

        self.evaluator = None
        if 'evaluator' in self.env_spec:
            params = deepcopy(ps.get(self.env_spec, 'evaluator'))
            EvaluatorClass = getattr(evaluator, params.pop('name'))
            self.evaluator = EvaluatorClass(**params) 

        self.simulator = UserSimulator(self.nlu, self.agenda, self.nlg)
        self.simulator.init_session()
        self.action_vocab = ActionVocab(num_actions=action_dim)
        self.history = []
        self.last_act = None

        self.stat = {'success':0, 'fail':0}

    def reset(self, train_mode, config):
        self.simulator.init_session()
        self.history = []
        user_response, user_act, session_over, reward = self.simulator.response("null", self.history)
        self.last_act = user_act
        logger.act(f'User action: {user_act}')
        self.history.extend(["null", f'{user_response}'])
        self.env_info = [State(user_response, 0., session_over)] 
        # update evaluator
        if self.evaluator:
            self.evaluator.add_goal(self.get_goal())
            logger.act(f'Goal: {self.get_goal()}')
        return self.env_info 

    def get_goal(self):
        return deepcopy(self.simulator.policy.domain_goals)

    def get_last_act(self):
        return deepcopy(self.last_act)

    def get_sys_act(self):
        return deepcopy(self.simulator.sys_act)

    def step(self, action):
        user_response, user_act, session_over, reward = self.simulator.response(action, self.history)
        self.last_act = user_act
        self.history.extend([f'{action}', f'{user_response}'])
        logger.act(f'Inferred system action: {self.get_sys_act()}')
        # update evaluator
        if self.evaluator:
            self.evaluator.add_sys_da(self.get_sys_act())
            self.evaluator.add_usr_da(self.get_last_act())
            if session_over:
                reward = 2.0 * self.simulator.policy.max_turn if self.evaluator.task_success() else -1.0 * self.simulator.policy.max_turn
            else:
                reward = -1.0
        self.env_info = [State(user_response, reward, session_over)] 
        return self.env_info 

    def rule_policy(self, state, algorithm, body):
        def find_best_delex_act(action):
            def _score(a1, a2):
                score = 0
                for domain_act in a1:
                    if domain_act not in a2:
                        score += len(a1[domain_act])
                    else:
                        score += len(set(a1[domain_act]) - set(a2[domain_act]))
                return score

            best_p_action_index = -1 
            best_p_score = math.inf 
            best_pn_action_index = -1 
            best_pn_score = math.inf 
            for i, v_action in enumerate(self.action_vocab.vocab):
                if v_action == action:
                    return i
                else:
                    p_score = _score(action, v_action)
                    n_score = _score(v_action, action)
                    if p_score > 0 and n_score == 0 and p_score < best_p_score:
                        best_p_action_index = i
                        best_p_score = p_score
                    else:
                        if p_score + n_score < best_pn_score:
                            best_pn_action_index = i
                            best_pn_score = p_score + n_score
            if best_p_action_index >= 0:
                return best_p_action_index
            return best_pn_action_index

        rule_act = self.sys_policy.predict(state)
        delex_act = {}
        for domain_act in rule_act:
            domain, act_type = domain_act.split('-', 1)
            if act_type in ['NoOffer', 'OfferBook']:
                delex_act[domain_act] = ['none'] 
            elif act_type in ['Select']:
                for sv in rule_act[domain_act]:
                    if sv[0] != "none":
                        delex_act[domain_act] = [sv[0]] 
                        break
            else:
                delex_act[domain_act] = [sv[0] for sv in rule_act[domain_act]] 
        action = find_best_delex_act(delex_act)

        return action 

    def close(self):
        pass


class MultiWozEnv(BaseEnv):
    '''
    Wrapper for Unity ML-Agents env to work with the Lab.

    e.g. env_spec
    "env": [{
      "name": "gridworld",
      "max_t": 20,
      "max_tick": 3,
      "unity": {
        "gridSize": 6,
        "numObstacles": 2,
        "numGoals": 1
      }
    }],
    '''

    def __init__(self, spec, e=None):
        super(MultiWozEnv, self).__init__(spec, e)
        self.action_dim = self.observation_dim = 0
        util.set_attr(self, self.env_spec, [
            'observation_dim',
            'action_dim',
        ])
        worker_id = int(f'{os.getpid()}{self.e+int(ps.unique_id())}'[-4:])
        self.u_env = MultiWozEnvironment(self.env_spec, worker_id, self.action_dim)
        self.evaluator = self.u_env.evaluator
        self.patch_gym_spaces(self.u_env)
        self._set_attr_from_u_env(self.u_env)

        logger.info(util.self_desc(self))

    def patch_gym_spaces(self, u_env):
        '''
        For standardization, use gym spaces to represent observation and action spaces.
        This method iterates through the multiple brains (multiagent) then constructs and returns lists of observation_spaces and action_spaces
        '''
        observation_shape = (self.observation_dim,)
        observation_space = spaces.Box(low=0, high=1, shape=observation_shape, dtype=np.int32)
        set_gym_space_attr(observation_space)
        action_space = spaces.Discrete(self.action_dim)
        set_gym_space_attr(action_space)
        # set for singleton
        u_env.observation_space = observation_space
        u_env.action_space = action_space

    def _get_env_info(self, env_info_dict, a):
        ''''''
        return self.u_env.env_info[a]

    @lab_api
    def reset(self):
        # _reward = np.nan
        env_info_dict = self.u_env.reset(train_mode=(util.get_lab_mode() != 'dev'), config=self.env_spec.get('multiwoz'))
        a, b = 0, 0  # default singleton aeb
        env_info_a = self._get_env_info(env_info_dict, a)
        state = env_info_a.states[b]
        self.done = False
        logger.debug(f'Env {self.e} reset state: {state}')
        return state

    @lab_api
    def step(self, action):
        env_info_dict = self.u_env.step(action)
        a, b = 0, 0  # default singleton aeb
        env_info_a = self._get_env_info(env_info_dict, a)
        reward = env_info_a.rewards[b]  # * self.reward_scale
        state = env_info_a.states[b]
        done = env_info_a.local_done[b]
        self.done = done = done or self.clock.t > self.max_t
        logger.debug(f'Env {self.e} step reward: {reward}, state: {state}, done: {done}')
        return state, reward, done, env_info_a 

    @lab_api
    def close(self):
        self.u_env.close()

    def get_goal(self):
        return self.u_env.get_goal()

    def get_last_act(self):
        return self.u_env.get_last_act()

    def get_sys_act(self):
        return self.u_env.get_sys_act()

    def get_task_success(self):
        return self.u_env.simulator.policy.goal.task_complete()