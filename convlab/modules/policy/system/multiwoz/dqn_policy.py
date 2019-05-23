'''
DQN Agent
'''


import random, copy, json, pickle
import numpy as np

from .dialog_config import *
from .rule_based_multiwoz_bot import *
from convlab.modules.policy.system.policy import SysPolicy
from .qlearning import DQN
from convlab.modules.util.multiwoz_slot_trans import REF_SYS_DA
from .rule_based_multiwoz_bot import generate_car


class DQNPolicy(SysPolicy):
    def __init__(self, act_types, slots, slot_val_dict=None, params=None):
        """
        Constructor for Rule_Based_Sys_Policy class.
        Args:
            act_types (list): A list of dialog acts.
            slots (list): A list of slot names.
            slot_dict (dict): Map slot name to its value set.
        """
        SysPolicy.__init__(self)

        self.act_types = act_types
        self.slots = slots
        self.slot_dict = slot_val_dict
        #self._build_model()
        
        self.act_cardinality = len(act_types)
        self.slot_cardinality = len(slots)
        
        self.feasible_actions = feasible_actions
        self.num_actions = len(self.feasible_actions)
        
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.experience_replay_pool = [] #experience replay pool <s_t, a_t, r_t, s_t+1>
        
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)
        
        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        
        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)
        self.clone_dqn = copy.deepcopy(self.dqn)
        
        self.cur_bellman_err = 0
                
        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.dqn.model = copy.deepcopy(self.load_trained_DQN(params['trained_model_path']))
            self.clone_dqn = copy.deepcopy(self.dqn)
            self.predict_mode = True
            self.warm_start = 2
    
    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Please check util/state.py
        Returns:
            action (list): System act, in the form of {act_type1: [[slot_name_1, value_1], [slot_name_2, value_2], ...], ...}
        """
        
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return act_slot_response
    
    def init_session(self):
        """
        Initialize one session
        """
        self.cur_inform_slot_id = 0
        self.cur_request_slot_id = 0
        self.domains = ['Taxi']
        
        
    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """
        
        self.current_slot_id = 0
        self.phase = 0
        
        self.current_request_slot_id = 0
        self.current_inform_slot_id = 0
        
        #self.request_set = dialog_config.movie_request_slots #['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
        
    def initialize_config(self, req_set, inf_set):
        """ Initialize request_set and inform_set """
        
        self.request_set = req_set
        self.inform_set = inf_set
        self.current_request_slot_id = 0
        self.current_inform_slot_id = 0
        
    def state_to_action(self, state):
        """ DQN: Input state, output action """
        
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
        
    
    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        # state info
        user_action = state['user_action']
        kb_results_dict = state['kb_results_dict']
        belief_state = state['belief_state']
        history = state['history'] # nl
        
        
        #current_slots = state['current_slots']
        #agent_last = state['agent_action'] # missing
        
        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep =  np.zeros((1, self.act_cardinality))
        for key in user_action:
            if key in self.act_types:
                act_index = self.act_types.index(key)
                #user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0
                user_act_rep[0, act_index] = 1.0
        
        ########################################################################
        #     Create bag of inform & request slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for key in user_action: 
            s_v_pairs = user_action[key]
            for s_v in s_v_pairs:
                s = s_v[0]
                v = s_v[1]
                
                if v == "?": # request
                    if s in self.slots:
                        s_i = self.slots.index(s)
                        user_request_slots_rep[0, s_i] = 1.0
                else:
                    if s in self.slots:
                        s_i = self.slots.index(s)
                        user_inform_slots_rep[0, s_i] = 1.0
        
        #for slot in user_action['inform_slots'].keys():
        #    user_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        #user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        #for slot in user_action['request_slots'].keys():
        #    user_request_slots_rep[0, self.slot_set[slot]] = 1.0
        
        ## change here
        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        print('belief_state', json.dumps(belief_state, indent=2))
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        belief_state_keys = ['book', 'semi']
        for domain in belief_state:
            if domain == "inform_slots": continue
            for bs_key in belief_state_keys:
                for s in belief_state[domain]['book']:
                    v = belief_state[domain]['book'][s]
                    if len(v) != 0:
                        if s in self.slots:
                            s_i = self.slots.index(s)
                            current_slots_rep[0, s_i] = 1.0
        
        #for slot in current_slots['inform_slots']:
        #    current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1,self.act_cardinality))
        #if agent_last:
        #    agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0
        
        
        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        #if agent_last:
        #    for slot in agent_last['inform_slots'].keys():
        #        agent_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        #if agent_last:
        #    for slot in agent_last['request_slots'].keys():
        #        agent_request_slots_rep[0,self.slot_set[slot]] = 1.0
        
        #turn_rep = np.zeros((1,1)) + state['turn'] / 10.
        turn_rep = np.zeros((1,1))
        
        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        #turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        #kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))
        #for slot in kb_results_dict:
        #    if slot in self.slot_set:
        #        kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        #kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))
        #for slot in kb_results_dict:
        #    if slot in self.slot_set:
        #        kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep, 
                                               agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation
      
      
    def run_policy(self, representation):
        """ epsilon-greedy policy """
        
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_request_inform_policy() 
                #return self.rule_policy()
            else:
                return self.dqn.predict(representation, {}, predict_model=True)
    
    def rule_policy(self):
        """ Rule Policy """
        
        domain = "Taxi"
        if self.current_slot_id < len(REF_SYS_DA[domain]):
            key = list(REF_SYS_DA[domain])[self.current_slot_id]
            slot = REF_SYS_DA[domain][key]
            
            diaact = domain + "-Inform"
            val = generate_car()
                    
            act_slot_response[diaact] = []
            act_slot_response[diaact].append([slot, val])
            
            self.current_slot_id += 1
        else:
            act_slot_response['general-bye'] = []
            self.current_slot_id = 0
        
           
        # old
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {} }
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {} }
                
        return self.action_index(act_slot_response)
    
    def rule_request_inform_policy(self):
        """ Rule Request and Inform Policy """
        
        act_slot_response = {}
        domain = self.domains[0]        
            
        if self.cur_inform_slot_id < len(REF_SYS_DA[domain]):
            key = list(REF_SYS_DA[domain])[self.cur_inform_slot_id]
            slot = REF_SYS_DA[domain][key]
            
            diaact = domain + "-Inform"
            val = generate_car()
                    
            act_slot_response[diaact] = []
            act_slot_response[diaact].append([slot, val])
            
            self.cur_inform_slot_id += 1
        elif self.cur_request_slot_id < len(REF_SYS_DA[domain]):
            key = list(REF_SYS_DA[domain])[self.cur_request_slot_id]
            slot = REF_SYS_DA[domain][key]
            
            diaact = domain + "-Request"
            val = "?"
                    
            act_slot_response[diaact] = []
            act_slot_response[diaact].append([slot, val])
            
            self.cur_request_slot_id += 1
        else:
            act_slot_response['general-bye'] = []
            self.cur_request_slot_id = 0
            self.cur_inform_slot_id = 0
            
        #return act_slot_response
        return self.action_index(act_slot_response)
        
    
    def action_index(self, act_slot_response):
        """ Return the index of action """
        
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
            
        print(act_slot_response)
        #raise Exception("action index not found")
        #return None
        return 1 # default
    
    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register feedback from the environment, to be stored as future training data """
        
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)
        
        if self.predict_mode == False: # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else: # Prediction Mode
            self.experience_replay_pool.append(training_example)
    
    def train(self, batch_size=1, num_batches=100):
        """ Train DQN with experience replay """
        
        for iter_batch in range(num_batches):
            self.cur_bellman_err = 0
            for iter in range(len(self.experience_replay_pool)/(batch_size)):
                batch = [random.choice(self.experience_replay_pool) for i in xrange(batch_size)]
                batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma}, self.clone_dqn)
                self.cur_bellman_err += batch_struct['cost']['total_cost']
            
            print("cur bellman err %.4f, experience replay pool %s" % (float(self.cur_bellman_err)/len(self.experience_replay_pool), len(self.experience_replay_pool)))
            
            
    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """
        
        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print('saved model in %s' % (path, ))
        except Exception as e:
            print('Error: Writing model fails: %s' % (path, ))
            print(e)  
    
    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""
        
        self.experience_replay_pool = pickle.load(open(path, 'rb'))
    
             
    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """
        
        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']
        
        print("trained DQN Parameters:")
        print(json.dumps(trained_file['params'], indent=2))
        return model