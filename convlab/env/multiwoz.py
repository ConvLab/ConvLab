# Modified by Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import math
import os

import numpy as np
import pydash as ps
from gym import spaces

from convlab.env.base import BaseEnv, ENV_DATA_NAMES, set_gym_space_attr
# from convlab.env.registration import get_env_path
from convlab.lib import logger, util
from convlab.lib.decorator import lab_api
from convlab import UserPolicyAgendaMultiWoz
from convlab import UserSimulator
# from convlab.modules.policy.system.util import action_decoder, ActionVocab, state_encoder
from convlab.modules.policy.system.multiwoz.rule_based_multiwoz_bot import RuleBasedMultiwozBot
from convlab.modules.action_decoder.multiwoz.multiwoz_vocab_action_decoder import ActionVocab
import convlab.modules.policy.user.multiwoz as user_policy
import convlab.modules.policy.system.multiwoz as sys_policy
import convlab.modules.nlu.multiwoz as nlu
import convlab.modules.nlg.multiwoz as nlg

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

        self.simulator = UserSimulator(self.nlu, self.agenda, self.nlg)
        self.simulator.init_session()
        self.action_vocab = ActionVocab(num_actions=action_dim)
        self.history = []

        self.stat = {'success':0, 'fail':0}

    def reset(self, train_mode, config):
        self.simulator.init_session()
        self.history = []
        user_response, user_act, session_over, reward = self.simulator.response("null", self.history)
        str_user_response = '{}'.format(user_response)
        self.history.extend(["null", str_user_response])
        self.env_info = [State(user_response, 0., session_over)] 
        return self.env_info 

    def step(self, action):
        user_response, user_act, session_over, reward = self.simulator.response(action, self.history)
        str_sys_response = '{}'.format(action)
        str_user_response = '{}'.format(user_response)
        self.history.extend([str_sys_response, str_user_response])
        if session_over:
            dialog_status = self.simulator.policy.goal.task_complete()
            if dialog_status:
                self.stat['success'] += 1
            else: self.stat['fail'] += 1
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
        # pprint(delex_act)
        action = find_best_delex_act(delex_act)

        return action 

    def close(self):
        print('\nstatistics: %s' % (self.stat))
        print('\nsuccess rate: %s' % (self.stat['success']/(self.stat['success']+self.stat['fail'])))


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

    def __init__(self, spec, e=None, env_space=None):
        super(MultiWozEnv, self).__init__(spec, e, env_space)
        self.action_dim = self.observation_dim = 0
        util.set_attr(self, self.env_spec, [
            'observation_dim',
            'action_dim',
        ])
        worker_id = int(f'{os.getpid()}{self.e+int(ps.unique_id())}'[-4:])
        self.u_env = MultiWozEnvironment(self.env_spec, worker_id, self.action_dim)
        self.patch_gym_spaces(self.u_env)
        self._set_attr_from_u_env(self.u_env)
        # assert self.max_t is not None
        if env_space is None:  # singleton mode
            pass
        else:
            self.space_init(env_space)

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
        _reward = np.nan
        env_info_dict = self.u_env.reset(train_mode=(util.get_lab_mode() != 'dev'), config=self.env_spec.get('multiwoz'))
        a, b = 0, 0  # default singleton aeb
        env_info_a = self._get_env_info(env_info_dict, a)
        state = env_info_a.states[b]
        self.done = done = False
        logger.debug(f'Env {self.e} reset reward: {_reward}, state: {state}, done: {done}')
        return _reward, state, done

    @lab_api
    def step(self, action):
        env_info_dict = self.u_env.step(action)
        a, b = 0, 0  # default singleton aeb
        env_info_a = self._get_env_info(env_info_dict, a)
        reward = env_info_a.rewards[b] * self.reward_scale
        state = env_info_a.states[b]
        done = env_info_a.local_done[b]
        self.done = done = done or self.clock.t > self.max_t
        logger.debug(f'Env {self.e} step reward: {reward}, state: {state}, done: {done}')
        return reward, state, done

    @lab_api
    def close(self):
        self.u_env.close()

    # NOTE optional extension for multi-agent-env

    @lab_api
    def space_init(self, env_space):
        '''Post init override for space env. Note that aeb is already correct from __init__'''
        self.env_space = env_space
        self.aeb_space = env_space.aeb_space
        self.observation_spaces = [self.observation_space]
        self.action_spaces = [self.action_space]

    @lab_api
    def space_reset(self):
        self._check_u_brain_to_agent()
        self.done = False
        env_info_dict = self.u_env.reset(train_mode=(util.get_lab_mode() != 'dev'), config=self.env_spec.get('multiwoz'))
        _reward_e, state_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for (a, b), body in util.ndenumerate_nonan(self.body_e):
            env_info_a = self._get_env_info(env_info_dict, a)
            self._check_u_agent_to_body(env_info_a, a)
            state = env_info_a.states[b]
            state_e[(a, b)] = state
            done_e[(a, b)] = self.done
        logger.debug(f'Env {self.e} reset reward_e: {_reward_e}, state_e: {state_e}, done_e: {done_e}')
        return _reward_e, state_e, done_e

    @lab_api
    def space_step(self, action_e):
        # TODO implement clock_speed: step only if self.clock.to_step()
        if self.done:
            return self.space_reset()
        action_e = util.nanflatten(action_e)
        env_info_dict = self.u_env.step(action_e)
        reward_e, state_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for (a, b), body in util.ndenumerate_nonan(self.body_e):
            env_info_a = self._get_env_info(env_info_dict, a)
            reward_e[(a, b)] = env_info_a.rewards[b] * self.reward_scale
            state_e[(a, b)] = env_info_a.states[b]
            done_e[(a, b)] = env_info_a.local_done[b]
        self.done = (util.nonan_all(done_e) or self.clock.t > self.max_t)
        logger.debug(f'Env {self.e} step reward_e: {reward_e}, state_e: {state_e}, done_e: {done_e}')
        return reward_e, state_e, done_e
