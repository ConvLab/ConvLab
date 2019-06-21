# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pydash as ps
from gym import spaces

from convlab.env.base import BaseEnv, ENV_DATA_NAMES, set_gym_space_attr
# from convlab.env.registration import get_env_path
from convlab.lib import logger, util
from convlab.lib.decorator import lab_api

logger = logger.get_logger(__name__)


################################################################################
#   Parameters for Agents
################################################################################
agent_params = {}
agent_params['max_turn'] = 40 
agent_params['agent_run_mode'] = 1 
agent_params['agent_act_level'] = 0 


################################################################################
#   Parameters for User Simulators
################################################################################
usersim_params = {}
usersim_params['max_turn'] = 40 
usersim_params['slot_err_probability'] = 0
usersim_params['slot_err_mode'] = 0 
usersim_params['intent_err_probability'] = 0 
usersim_params['simulator_run_mode'] = 1 
usersim_params['simulator_act_level'] = 0
usersim_params['learning_phase'] = 'all' 

DATAPATH=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/movie")

dict_path = os.path.join(DATAPATH, 'dicts.v3.p') 
goal_file_path = os.path.join(DATAPATH, 'user_goals_first_turn_template.part.movie.v1.p')

# load the user goals from .p file
all_goal_set = pickle.load(open(goal_file_path, 'rb'))

# split goal set
split_fold = 5
goal_set = {'train':[], 'valid':[], 'test':[], 'all':[]}
for u_goal_id, u_goal in enumerate(all_goal_set):
    if u_goal_id % split_fold == 1: goal_set['test'].append(u_goal)
    else: goal_set['train'].append(u_goal)
    goal_set['all'].append(u_goal)
# end split goal set

movie_kb_path = os.path.join(DATAPATH, 'movie_kb.1k.p')
# movie_kb = pickle.load(open(movie_kb_path, 'rb'), encoding='latin1')
movie_dictionary = pickle.load(open(movie_kb_path, 'rb'), encoding='latin1')

def text_to_dict(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers) """
    
    slot_set = {}
    with open(path, 'r') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
    return slot_set

act_set = text_to_dict(os.path.join(DATAPATH, 'dia_acts.txt')) 
slot_set = text_to_dict(os.path.join(DATAPATH, 'slot_set.txt'))

################################################################################
# a movie dictionary for user simulator - slot:possible values
################################################################################
# movie_dictionary = pickle.load(open(dict_path, 'rb'))

sys_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids']
sys_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids', 'taskcomplete', 'ticket']

start_dia_acts = {
    #'greeting':[],
    'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople']
}

################################################################################
# Dialog status
################################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

# Rewards
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = 0

################################################################################
#  Special Slot Values
################################################################################
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"
TICKET_AVAILABLE = 'Ticket Available'

################################################################################
#  Constraint Check
################################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

################################################################################
#  NLG Beam Search
################################################################################
nlg_beam_size = 10

################################################################################
#  run_mode: 0 for dia-act; 1 for NL; 2 for no output
################################################################################
run_mode = 3
auto_suggest = 0

################################################################################
#   A Basic Set of Feasible actions to be Consdered By an RL agent
################################################################################
feasible_actions = [
    ############################################################################
    #   greeting actions
    ############################################################################
    #{'diaact':"greeting", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   confirm_question actions
    ############################################################################
    {'diaact':"confirm_question", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   confirm_answer actions
    ############################################################################
    {'diaact':"confirm_answer", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   thanks actions
    ############################################################################
    {'diaact':"thanks", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   deny actions
    ############################################################################
    {'diaact':"deny", 'inform_slots':{}, 'request_slots':{}},
]
############################################################################
#   Adding the inform actions
############################################################################
for slot in sys_inform_slots:
    feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})

############################################################################
#   Adding the request actions
############################################################################
for slot in sys_request_slots:
    feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})


class UserSimulator:
    """ Parent class for all user sims to inherit from """

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """
        
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        
        self.max_turn = usersim_params['max_turn']
        self.slot_err_probability = usersim_params['slot_err_probability']
        self.slot_err_mode = usersim_params['slot_err_mode']
        self.intent_err_probability = usersim_params['intent_err_probability']
        

    def initialize_episode(self):
        """ Initialize a new episode (dialog)"""

        print("initialize episode called, generating goal")
        self.goal =  random.choice(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        episode_over, user_action = self._sample_action()
        assert (episode_over != 1),' but we just started'
        return user_action


    def next(self, system_action):
        pass
    
    
    
    def set_nlg_model(self, nlg_model):
        self.nlg_model = nlg_model  
    
    def set_nlu_model(self, nlu_model):
        self.nlu_model = nlu_model
    
    
    
    def add_nl_to_action(self, user_action):
        """ Add NL to User Dia_Act """
        
        user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(user_action, 'usr')
        user_action['nl'] = user_nlg_sentence
        
        if self.simulator_act_level == 1:
            user_nlu_res = self.nlu_model.generate_dia_act(user_action['nl']) # NLU
            if user_nlu_res != None:
                #user_nlu_res['diaact'] = user_action['diaact'] # or not?
                user_action.update(user_nlu_res)



class RuleSimulator(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """
    
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """
        
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        
        self.max_turn = usersim_params['max_turn']
        self.slot_err_probability = usersim_params['slot_err_probability']
        self.slot_err_mode = usersim_params['slot_err_mode']
        self.intent_err_probability = usersim_params['intent_err_probability']
        
        self.simulator_run_mode = usersim_params['simulator_run_mode']
        self.simulator_act_level = usersim_params['simulator_act_level']
        
        self.learning_phase = usersim_params['learning_phase']
    
    def initialize_episode(self):
        """ Initialize a new episode (dialog) 
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """
        
        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0
        
        self.episode_over = False
        self.dialog_status = NO_OUTCOME_YET
        
        #self.goal =  random.choice(self.start_set)
        self.goal = self._sample_goal(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        self.constraint_check = CONSTRAINT_CHECK_FAILURE
  
        """ Debug: build a fake goal mannually """
        #self.debug_falk_goal()
        
        # sample first action
        user_action = self._sample_action()
        assert (self.episode_over != 1),' but we just started'
        return user_action  
        
    def _sample_action(self):
        """ randomly sample a start action based on user goal """
        
        self.state['diaact'] = random.choice(list(start_dia_acts.keys()))
        
        # "sample" informed slots
        if len(self.goal['inform_slots']) > 0:
            known_slot = random.choice(list(self.goal['inform_slots'].keys()))
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

            if 'moviename' in self.goal['inform_slots']: # 'moviename' must appear in the first user turn
                self.state['inform_slots']['moviename'] = self.goal['inform_slots']['moviename']
                
            for slot in self.goal['inform_slots'].keys():
                if known_slot == slot or slot == 'moviename': continue
                self.state['rest_slots'].append(slot)
        
        self.state['rest_slots'].extend(self.goal['request_slots'].keys())
        
        # "sample" a requested slot
        request_slot_set = list(self.goal['request_slots'].keys())
        request_slot_set.remove('ticket')
        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
        else:
            request_slot = 'ticket'
        self.state['request_slots'][request_slot] = 'UNK'
        
        if len(self.state['request_slots']) == 0:
            self.state['diaact'] = 'inform'

        if (self.state['diaact'] in ['thanks','closing']): self.episode_over = True #episode_over = True
        else: self.episode_over = False #episode_over = False

        sample_action = {}
        sample_action['diaact'] = self.state['diaact']
        sample_action['inform_slots'] = self.state['inform_slots']
        sample_action['request_slots'] = self.state['request_slots']
        sample_action['turn'] = self.state['turn']
        
        # self.add_nl_to_action(sample_action)
        return sample_action
    
    def _sample_goal(self, goal_set):
        """ sample a user goal  """
        
        sample_goal = random.choice(self.start_set[self.learning_phase])
        return sample_goal
    
    
    def corrupt(self, user_action):
        """ Randomly corrupt an action with error probs (slot_err_probability and slot_err_mode) on Slot and Intent (intent_err_probability). """
        
        for slot in user_action['inform_slots'].keys():
            slot_err_prob_sample = random.random()
            if slot_err_prob_sample < self.slot_err_probability: # add noise for slot level
                if self.slot_err_mode == 0: # replace the slot_value only
                    if slot in self.movie_dict.keys(): user_action['inform_slots'][slot] = random.choice(self.movie_dict[slot])
                elif self.slot_err_mode == 1: # combined
                    slot_err_random = random.random()
                    if slot_err_random <= 0.33:
                        if slot in self.movie_dict.keys(): user_action['inform_slots'][slot] = random.choice(self.movie_dict[slot])
                    elif slot_err_random > 0.33 and slot_err_random <= 0.66:
                        del user_action['inform_slots'][slot]
                        random_slot = random.choice(self.movie_dict.keys())
                        user_action[random_slot] = random.choice(self.movie_dict[random_slot])
                    else:
                        del user_action['inform_slots'][slot]
                elif self.slot_err_mode == 2: #replace slot and its values
                    del user_action['inform_slots'][slot]
                    random_slot = random.choice(self.movie_dict.keys())
                    user_action[random_slot] = random.choice(self.movie_dict[random_slot])
                elif self.slot_err_mode == 3: # delete the slot
                    del user_action['inform_slots'][slot]
                    
        intent_err_sample = random.random()
        if intent_err_sample < self.intent_err_probability: # add noise for intent level
            user_action['diaact'] = random.choice(self.act_set.keys())
    
    def debug_falk_goal(self):
        """ Debug function: build a fake goal mannually (Can be moved in future) """
        
        self.goal['inform_slots'].clear()
        #self.goal['inform_slots']['city'] = 'seattle'
        self.goal['inform_slots']['numberofpeople'] = '2'
        #self.goal['inform_slots']['theater'] = 'amc pacific place 11 theater'
        #self.goal['inform_slots']['starttime'] = '10:00 pm'
        #self.goal['inform_slots']['date'] = 'tomorrow'
        self.goal['inform_slots']['moviename'] = 'zoology'
        self.goal['inform_slots']['distanceconstraints'] = 'close to 95833'
        self.goal['request_slots'].clear()
        self.goal['request_slots']['ticket'] = 'UNK'
        self.goal['request_slots']['theater'] = 'UNK'
        self.goal['request_slots']['starttime'] = 'UNK'
        self.goal['request_slots']['date'] = 'UNK'
        
    def next(self, system_action):
        """ Generate next User Action based on last System Action """
        
        self.state['turn'] += 2
        self.episode_over = False
        self.dialog_status = NO_OUTCOME_YET
        
        sys_act = system_action['diaact']
        
        if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
            self.dialog_status = FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
        else:
            self.state['history_slots'].update(self.state['inform_slots'])
            self.state['inform_slots'].clear()

            if sys_act == "inform":
                self.response_inform(system_action)
            elif sys_act == "multiple_choice":
                self.response_multiple_choice(system_action)
            elif sys_act == "request":
                self.response_request(system_action) 
            elif sys_act == "thanks":
                self.response_thanks(system_action)
            elif sys_act == "confirm_answer":
                self.response_confirm_answer(system_action)
            elif sys_act == "closing":
                self.episode_over = True
                self.state['diaact'] = "thanks"

        self.corrupt(self.state)
        
        response_action = {}
        response_action['diaact'] = self.state['diaact']
        response_action['inform_slots'] = self.state['inform_slots']
        response_action['request_slots'] = self.state['request_slots']
        response_action['turn'] = self.state['turn']
        response_action['nl'] = ""
        
        # add NL to dia_act
        # self.add_nl_to_action(response_action)                       
        return response_action, self.episode_over, self.dialog_status
    
    
    def response_confirm_answer(self, system_action):
        """ Response for Confirm_Answer (System Action) """
    
        if len(self.state['rest_slots']) > 0:
            request_slot = random.choice(self.state['rest_slots'])

            if request_slot in self.goal['request_slots'].keys():
                self.state['diaact'] = "request"
                self.state['request_slots'][request_slot] = "UNK"
            elif request_slot in self.goal['inform_slots'].keys():
                self.state['diaact'] = "inform"
                self.state['inform_slots'][request_slot] = self.goal['inform_slots'][request_slot]
                if request_slot in self.state['rest_slots']:
                    self.state['rest_slots'].remove(request_slot)
        else:
            self.state['diaact'] = "thanks"
            
    def response_thanks(self, system_action):
        """ Response for Thanks (System Action) """
        
        self.episode_over = True
        self.dialog_status = SUCCESS_DIALOG

        request_slot_set = deepcopy(list(self.state['request_slots'].keys()))
        if 'ticket' in request_slot_set:
            request_slot_set.remove('ticket')
        rest_slot_set = deepcopy(self.state['rest_slots'])
        if 'ticket' in rest_slot_set:
            rest_slot_set.remove('ticket')

        if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
            self.dialog_status = FAILED_DIALOG

        for info_slot in self.state['history_slots'].keys():
            if self.state['history_slots'][info_slot] == NO_VALUE_MATCH:
                self.dialog_status = FAILED_DIALOG
            if info_slot in self.goal['inform_slots'].keys():
                if self.state['history_slots'][info_slot] != self.goal['inform_slots'][info_slot]:
                    self.dialog_status = FAILED_DIALOG

        if 'ticket' in system_action['inform_slots'].keys():
            if system_action['inform_slots']['ticket'] == NO_VALUE_MATCH:
                self.dialog_status = FAILED_DIALOG
                
        if self.constraint_check == CONSTRAINT_CHECK_FAILURE:
            self.dialog_status = FAILED_DIALOG
    
    def response_request(self, system_action):
        """ Response for Request (System Action) """
        
        if len(system_action['request_slots'].keys()) > 0:
            slot = list(system_action['request_slots'].keys())[0] # only one slot
            if slot in self.goal['inform_slots']: # request slot in user's constraints  #and slot not in self.state['request_slots'].keys():
                self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                self.state['diaact'] = "inform"
                if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                if slot in self.state['request_slots']: del self.state['request_slots'][slot]
                self.state['request_slots'].clear()
            elif slot in self.goal['request_slots'] and slot not in self.state['rest_slots'] and slot in self.state['history_slots']: # the requested slot has been answered
                self.state['inform_slots'][slot] = self.state['history_slots'][slot]
                self.state['request_slots'].clear()
                self.state['diaact'] = "inform"
            elif slot in self.goal['request_slots'].keys() and slot in self.state['rest_slots']: # request slot in user's goal's request slots, and not answered yet
                self.state['diaact'] = "request" # "confirm_question"
                self.state['request_slots'][slot] = "UNK"

                ########################################################################
                # Inform the rest of informable slots
                ########################################################################
                for info_slot in self.state['rest_slots']:
                    if info_slot in self.goal['inform_slots'].keys():
                        self.state['inform_slots'][info_slot] = self.goal['inform_slots'][info_slot]

                for info_slot in self.state['inform_slots'].keys():
                    if info_slot in self.state['rest_slots']:
                        self.state['rest_slots'].remove(info_slot)
            else:
                if len(self.state['request_slots']) == 0 and len(self.state['rest_slots']) == 0:
                    self.state['diaact'] = "thanks"
                else:
                    self.state['diaact'] = "inform"
                self.state['inform_slots'][slot] = I_DO_NOT_CARE
        else: # this case should not appear
            if len(self.state['rest_slots']) > 0:
                random_slot = random.choice(self.state['rest_slots'])
                if random_slot in self.goal['inform_slots'].keys():
                    self.state['inform_slots'][random_slot] = self.goal['inform_slots'][random_slot]
                    self.state['rest_slots'].remove(random_slot)
                    self.state['diaact'] = "inform"
                elif random_slot in self.goal['request_slots'].keys():
                    self.state['request_slots'][random_slot] = self.goal['request_slots'][random_slot]
                    self.state['diaact'] = "request"

    def response_multiple_choice(self, system_action):
        """ Response for Multiple_Choice (System Action) """
        
        slot = system_action['inform_slots'].keys()[0]
        if slot in self.goal['inform_slots'].keys():
            self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
        elif slot in self.goal['request_slots'].keys():
            self.state['inform_slots'][slot] = random.choice(system_action['inform_slots'][slot])

        self.state['diaact'] = "inform"
        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
        if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]
        
    def response_inform(self, system_action):
        """ Response for Inform (System Action) """
        
        if 'taskcomplete' in system_action['inform_slots'].keys(): # check all the constraints from agents with user goal
            self.state['diaact'] = "thanks"
            #if 'ticket' in self.state['rest_slots']: self.state['request_slots']['ticket'] = 'UNK'
            self.constraint_check = CONSTRAINT_CHECK_SUCCESS
                    
            if system_action['inform_slots']['taskcomplete'] == NO_VALUE_MATCH:
                self.state['history_slots']['ticket'] = NO_VALUE_MATCH
                if 'ticket' in self.state['rest_slots']: self.state['rest_slots'].remove('ticket')
                if 'ticket' in self.state['request_slots'].keys(): del self.state['request_slots']['ticket']
                    
            for slot in self.goal['inform_slots'].keys():
                #  Deny, if the answers from agent can not meet the constraints of user
                if slot not in system_action['inform_slots'].keys() or (self.goal['inform_slots'][slot].lower() != system_action['inform_slots'][slot].lower()):
                    self.state['diaact'] = "deny"
                    self.state['request_slots'].clear()
                    self.state['inform_slots'].clear()
                    self.constraint_check = CONSTRAINT_CHECK_FAILURE
                    break
        else:
            for slot in system_action['inform_slots'].keys():
                self.state['history_slots'][slot] = system_action['inform_slots'][slot]
                        
                if slot in self.goal['inform_slots'].keys():
                    if system_action['inform_slots'][slot] == self.goal['inform_slots'][slot]:
                        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                                
                        if len(self.state['request_slots']) > 0:
                            self.state['diaact'] = "request"
                        elif len(self.state['rest_slots']) > 0:
                            rest_slot_set = deepcopy(self.state['rest_slots'])
                            if 'ticket' in rest_slot_set:
                                rest_slot_set.remove('ticket')

                            if len(rest_slot_set) > 0:
                                inform_slot = random.choice(rest_slot_set) # self.state['rest_slots']
                                if inform_slot in self.goal['inform_slots'].keys():
                                    self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                                    self.state['diaact'] = "inform"
                                    self.state['rest_slots'].remove(inform_slot)
                                elif inform_slot in self.goal['request_slots'].keys():
                                    self.state['request_slots'][inform_slot] = 'UNK'
                                    self.state['diaact'] = "request"
                            else:
                                self.state['request_slots']['ticket'] = 'UNK'
                                self.state['diaact'] = "request"
                        else: # how to reply here?
                            self.state['diaact'] = "thanks" # replies "closing"? or replies "confirm_answer"
                    else: # != value  Should we deny here or ?
                        ########################################################################
                        # TODO When agent informs(slot=value), where the value is different with the constraint in user goal, Should we deny or just inform the correct value?
                        ########################################################################
                        self.state['diaact'] = "inform"
                        self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                else:
                    if slot in self.state['rest_slots']:
                        self.state['rest_slots'].remove(slot)
                    if slot in self.state['request_slots'].keys():
                        del self.state['request_slots'][slot]

                    if len(self.state['request_slots']) > 0:
                        request_set = list(self.state['request_slots'].keys())
                        if 'ticket' in request_set:
                            request_set.remove('ticket')

                        if len(request_set) > 0:
                            request_slot = random.choice(request_set)
                        else:
                            request_slot = 'ticket'

                        self.state['request_slots'][request_slot] = "UNK"
                        self.state['diaact'] = "request"
                    elif len(self.state['rest_slots']) > 0:
                        rest_slot_set = deepcopy(self.state['rest_slots'])
                        if 'ticket' in rest_slot_set:
                            rest_slot_set.remove('ticket')

                        if len(rest_slot_set) > 0:
                            inform_slot = random.choice(rest_slot_set) #self.state['rest_slots']
                            if inform_slot in self.goal['inform_slots'].keys():
                                self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                                self.state['diaact'] = "inform"
                                self.state['rest_slots'].remove(inform_slot)
                                        
                                if 'ticket' in self.state['rest_slots']:
                                    self.state['request_slots']['ticket'] = 'UNK'
                                    self.state['diaact'] = "request"
                            elif inform_slot in self.goal['request_slots'].keys():
                                self.state['request_slots'][inform_slot] = self.goal['request_slots'][inform_slot]
                                self.state['diaact'] = "request"
                        else:
                            self.state['request_slots']['ticket'] = 'UNK'
                            self.state['diaact'] = "request"
                    else:
                        self.state['diaact'] = "thanks" # or replies "confirm_answer"


class StateTracker:
    """ The state tracker maintains a record of which request slots are filled and which inform slots are filled """

    def __init__(self, act_set, slot_set, movie_dictionary):
        """ constructor for statetracker takes movie knowledge base and initializes a new episode

        Arguments:
        act_set                 --  The set of all acts availavle
        slot_set                --  The total set of available slots
        movie_dictionary        --  A representation of all the available movies. Generally this object is accessed via the KBHelper class

        Class Variables:
        history_vectors         --  A record of the current dialog so far in vector format (act-slot, but no values)
        history_dictionaries    --  A record of the current dialog in dictionary format
        current_slots           --  A dictionary that keeps a running record of which slots are filled current_slots['inform_slots'] and which are requested current_slots['request_slots'] (but not filed)
        action_dimension        --  # TODO indicates the dimensionality of the vector representaiton of the action
        kb_result_dimension     --  A single integer denoting the dimension of the kb_results features.
        turn_count              --  A running count of which turn we are at in the present dialog
        """
        self.movie_dictionary = movie_dictionary
        self.initialize_episode()
        self.history_vectors = None
        self.history_dictionaries = None
        self.current_slots = None
        self.action_dimension = 10      # TODO REPLACE WITH REAL VALUE
        self.kb_result_dimension = 10   # TODO  REPLACE WITH REAL VALUE
        self.turn_count = 0
        self.kb_helper = KBHelper(movie_dictionary)
        

    def initialize_episode(self):
        """ Initialize a new episode (dialog), flush the current state and tracked slots """
        
        self.action_dimension = 10
        self.history_vectors = np.zeros((1, self.action_dimension))
        self.history_dictionaries = []
        self.turn_count = 0
        self.current_slots = {}
        
        self.current_slots['inform_slots'] = {}
        self.current_slots['request_slots'] = {}
        self.current_slots['proposed_slots'] = {}
        self.current_slots['agent_request_slots'] = {}


    def dialog_history_vectors(self):
        """ Return the dialog history (both user and agent actions) in vector representation """
        return self.history_vectors


    def dialog_history_dictionaries(self):
        """  Return the dictionary representation of the dialog history (includes values) """
        return self.history_dictionaries


    def kb_results_for_state(self):
        """ Return the information about the database results based on the currently informed slots """
        ########################################################################
        # TODO Calculate results based on current informed slots
        ########################################################################
        kb_results = self.kb_helper.database_results_for_agent(self.current_slots) # replace this with something less ridiculous
        # TODO turn results into vector (from dictionary)
        results = np.zeros((0, self.kb_result_dimension))
        return results
        

    def get_state_for_agent(self):
        """ Get the state representatons to send to agent """
        #state = {'user_action': self.history_dictionaries[-1], 'current_slots': self.current_slots, 'kb_results': self.kb_results_for_state()}
        state = {'user_action': self.history_dictionaries[-1], 'current_slots': self.current_slots, #'kb_results': self.kb_results_for_state(), 
                 'kb_results_dict':self.kb_helper.database_results_for_agent(self.current_slots), 'turn': self.turn_count, 'history': self.history_dictionaries, 
                 'agent_action': self.history_dictionaries[-2] if len(self.history_dictionaries) > 1 else None}
        return deepcopy(state)
    
    def get_suggest_slots_values(self, request_slots):
        """ Get the suggested values for request slots """
        
        suggest_slot_vals = {}
        if len(request_slots) > 0: 
            suggest_slot_vals = self.kb_helper.suggest_slot_values(request_slots, self.current_slots)
        
        return suggest_slot_vals
    
    def get_current_kb_results(self):
        """ get the kb_results for current state """
        kb_results = self.kb_helper.available_results_from_kb(self.current_slots)
        return kb_results
    
    
    def update(self, agent_action=None, user_action=None):
        """ Update the state based on the latest action """

        ########################################################################
        #  Make sure that the function was called properly
        ########################################################################
        assert(not (user_action and agent_action))
        assert(user_action or agent_action)

        ########################################################################
        #   Update state to reflect a new action by the agent.
        ########################################################################
        if agent_action:
            
            ####################################################################
            #   Handles the act_slot response (with values needing to be filled)
            ####################################################################
            if agent_action['act_slot_response']:
                response = deepcopy(agent_action['act_slot_response'])
                
                inform_slots = self.kb_helper.fill_inform_slots(response['inform_slots'], self.current_slots) # TODO this doesn't actually work yet, remove this warning when kb_helper is functional
                agent_action_values = {'turn': self.turn_count, 'speaker': "agent", 'diaact': response['diaact'], 'inform_slots': inform_slots, 'request_slots':response['request_slots']}
                
                agent_action['act_slot_response'].update({'diaact': response['diaact'], 'inform_slots': inform_slots, 'request_slots':response['request_slots'], 'turn':self.turn_count})
                
            elif agent_action['act_slot_value_response']:
                agent_action_values = deepcopy(agent_action['act_slot_value_response'])
                # print("Updating state based on act_slot_value action from agent")
                agent_action_values['turn'] = self.turn_count
                agent_action_values['speaker'] = "agent"
                
            ####################################################################
            #   This code should execute regardless of which kind of agent produced action
            ####################################################################
            for slot in agent_action_values['inform_slots'].keys():
                self.current_slots['proposed_slots'][slot] = agent_action_values['inform_slots'][slot]
                self.current_slots['inform_slots'][slot] = agent_action_values['inform_slots'][slot] # add into inform_slots
                if slot in self.current_slots['request_slots'].keys():
                    del self.current_slots['request_slots'][slot]

            for slot in agent_action_values['request_slots'].keys():
                if slot not in self.current_slots['agent_request_slots']:
                    self.current_slots['agent_request_slots'][slot] = "UNK"

            self.history_dictionaries.append(agent_action_values)
            current_agent_vector = np.ones((1, self.action_dimension))
            self.history_vectors = np.vstack([self.history_vectors, current_agent_vector])
                            
        ########################################################################
        #   Update the state to reflect a new action by the user
        ########################################################################
        elif user_action:
            
            ####################################################################
            #   Update the current slots
            ####################################################################
            for slot in user_action['inform_slots'].keys():
                self.current_slots['inform_slots'][slot] = user_action['inform_slots'][slot]
                if slot in self.current_slots['request_slots'].keys():
                    del self.current_slots['request_slots'][slot]

            for slot in user_action['request_slots'].keys():
                if slot not in self.current_slots['request_slots']:
                    self.current_slots['request_slots'][slot] = "UNK"
            
            self.history_vectors = np.vstack([self.history_vectors, np.zeros((1,self.action_dimension))])
            new_move = {'turn': self.turn_count, 'speaker': "user", 'request_slots': user_action['request_slots'], 'inform_slots': user_action['inform_slots'], 'diaact': user_action['diaact']}
            self.history_dictionaries.append(deepcopy(new_move))

        ########################################################################
        #   This should never happen if the asserts passed
        ########################################################################
        else:
            pass

        ########################################################################
        #   This code should execute after update code regardless of what kind of action (agent/user)
        ########################################################################
        self.turn_count += 1


class KBHelper:
    """ An assistant to fill in values for the agent (which knows about slots of values) """
    
    def __init__(self, movie_dictionary):
        """ Constructor for a KBHelper """
        
        self.movie_dictionary = movie_dictionary
        self.cached_kb = defaultdict(list)
        self.cached_kb_slot = defaultdict(list)


    def fill_inform_slots(self, inform_slots_to_be_filled, current_slots):
        """ Takes unfilled inform slots and current_slots, returns dictionary of filled informed slots (with values)

        Arguments:
        inform_slots_to_be_filled   --  Something that looks like {starttime:None, theater:None} where starttime and theater are slots that the agent needs filled
        current_slots               --  Contains a record of all filled slots in the conversation so far - for now, just use current_slots['inform_slots'] which is a dictionary of the already filled-in slots

        Returns:
        filled_in_slots             --  A dictionary of form {slot1:value1, slot2:value2} for each sloti in inform_slots_to_be_filled
        """
        
        kb_results = self.available_results_from_kb(current_slots)
        if auto_suggest == 1:
            print('Number of movies in KB satisfying current constraints: ', len(kb_results))

        filled_in_slots = {}
        if 'taskcomplete' in inform_slots_to_be_filled.keys():
            filled_in_slots.update(current_slots['inform_slots'])
        
        for slot in inform_slots_to_be_filled.keys():
            if slot == 'numberofpeople':
                if slot in current_slots['inform_slots'].keys():
                    filled_in_slots[slot] = current_slots['inform_slots'][slot]
                elif slot in inform_slots_to_be_filled.keys():
                    filled_in_slots[slot] = inform_slots_to_be_filled[slot]
                continue

            if slot == 'ticket' or slot == 'taskcomplete':
                filled_in_slots[slot] = TICKET_AVAILABLE if len(kb_results)>0 else NO_VALUE_MATCH
                continue
            
            if slot == 'closing': continue

            ####################################################################
            #   Grab the value for the slot with the highest count and fill it
            ####################################################################
            values_dict = self.available_slot_values(slot, kb_results)

            values_counts = [(v, values_dict[v]) for v in values_dict.keys()]
            if len(values_counts) > 0:
                filled_in_slots[slot] = sorted(values_counts, key = lambda x: -x[1])[0][0]
            else:
                filled_in_slots[slot] = NO_VALUE_MATCH #"NO VALUE MATCHES SNAFU!!!"
           
        return filled_in_slots


    def available_slot_values(self, slot, kb_results):
        """ Return the set of values available for the slot based on the current constraints """
        
        slot_values = {}
        for movie_id in kb_results.keys():
            if slot in kb_results[movie_id].keys():
                slot_val = kb_results[movie_id][slot]
                if slot_val in slot_values.keys():
                    slot_values[slot_val] += 1
                else: slot_values[slot_val] = 1
        return slot_values

    def available_results_from_kb(self, current_slots):
        """ Return the available movies in the movie_kb based on the current constraints """
        
        ret_result = []
        current_slots = current_slots['inform_slots']
        constrain_keys = current_slots.keys()

        constrain_keys = filter(lambda k : k != 'ticket' and \
                                           k != 'numberofpeople' and \
                                           k!= 'taskcomplete' and \
                                           k != 'closing' , constrain_keys)
        constrain_keys = [k for k in constrain_keys if current_slots[k] != I_DO_NOT_CARE]

        query_idx_keys = frozenset(current_slots.items())
        cached_kb_ret = self.cached_kb[query_idx_keys]

        cached_kb_length = len(cached_kb_ret) if cached_kb_ret != None else -1
        if cached_kb_length > 0:
            return dict(cached_kb_ret)
        elif cached_kb_length == -1:
            return dict([])

        # kb_results = copy.deepcopy(self.movie_dictionary)
        for id in self.movie_dictionary.keys():
            kb_keys = self.movie_dictionary[id].keys()
            if len(set(constrain_keys).union(set(kb_keys)) ^ (set(constrain_keys) ^ set(kb_keys))) == len(
                    constrain_keys):
                match = True
                for idx, k in enumerate(constrain_keys):
                    if str(current_slots[k]).lower() == str(self.movie_dictionary[id][k]).lower():
                        continue
                    else:
                        match = False
                if match:
                    self.cached_kb[query_idx_keys].append((id, self.movie_dictionary[id]))
                    ret_result.append((id, self.movie_dictionary[id]))

            # for slot in current_slots['inform_slots'].keys():
            #     if slot == 'ticket' or slot == 'numberofpeople' or slot == 'taskcomplete' or slot == 'closing': continue
            #     if current_slots['inform_slots'][slot] == dialog_config.I_DO_NOT_CARE: continue
            #
            #     if slot not in self.movie_dictionary[movie_id].keys():
            #         if movie_id in kb_results.keys():
            #             del kb_results[movie_id]
            #     else:
            #         if current_slots['inform_slots'][slot].lower() != self.movie_dictionary[movie_id][slot].lower():
            #             if movie_id in kb_results.keys():
            #                 del kb_results[movie_id]
            
        if len(ret_result) == 0:
            self.cached_kb[query_idx_keys] = None

        ret_result = dict(ret_result)
        return ret_result
    
    def available_results_from_kb_for_slots(self, inform_slots):
        """ Return the count statistics for each constraint in inform_slots """
        
        kb_results = {key:0 for key in inform_slots.keys()}
        kb_results['matching_all_constraints'] = 0
        
        query_idx_keys = frozenset(inform_slots.items())
        cached_kb_slot_ret = self.cached_kb_slot[query_idx_keys]

        if len(cached_kb_slot_ret) > 0:
            return cached_kb_slot_ret[0]

        for movie_id in self.movie_dictionary.keys():
            all_slots_match = 1
            for slot in inform_slots.keys():
                if slot == 'ticket' or inform_slots[slot] == I_DO_NOT_CARE:
                    continue

                if slot in self.movie_dictionary[movie_id]:
                # if slot in self.movie_dictionary[movie_id]:
                    if inform_slots[slot].lower() == self.movie_dictionary[movie_id][slot].lower():
                        kb_results[slot] += 1
                    else:
                        all_slots_match = 0
                else:
                    all_slots_match = 0
            kb_results['matching_all_constraints'] += all_slots_match

        self.cached_kb_slot[query_idx_keys].append(kb_results)
        return kb_results

    
    def database_results_for_agent(self, current_slots):
        """ A dictionary of the number of results matching each current constraint. The agent needs this to decide what to do next. """

        database_results ={} # { date:100, distanceconstraints:60, theater:30,  matching_all_constraints: 5}
        database_results = self.available_results_from_kb_for_slots(current_slots['inform_slots'])
        return database_results
    
    def suggest_slot_values(self, request_slots, current_slots):
        """ Return the suggest slot values """
        
        avail_kb_results = self.available_results_from_kb(current_slots)
        return_suggest_slot_vals = {}
        for slot in request_slots.keys():
            avail_values_dict = self.available_slot_values(slot, avail_kb_results)
            values_counts = [(v, avail_values_dict[v]) for v in avail_values_dict.keys()]
            
            if len(values_counts) > 0:
                return_suggest_slot_vals[slot] = []
                sorted_dict = sorted(values_counts, key = lambda x: -x[1])
                for k in sorted_dict: return_suggest_slot_vals[slot].append(k[0])
            else:
                return_suggest_slot_vals[slot] = []
        
        return return_suggest_slot_vals



class State(object):
    def __init__(self, state=None, reward=None, done=None):
        self.states = [state]
        self.rewards = [reward]
        self.local_done = [done]


class MovieActInActOutEnvironment(object):
    def __init__(self, worker_id=None):
        self.worker_id = worker_id
        self.act_set = act_set
        self.slot_set = slot_set
        self.movie_dict = movie_dictionary
        self.user = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
        self.state_tracker = StateTracker(act_set, slot_set, movie_dictionary)
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        self.feasible_actions = feasible_actions
        self.num_actions = len(self.feasible_actions)
        self.max_turn = agent_params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        print(self.num_actions)
        print(self.state_dimension)
        self.env_info = [State()] 
        self.stat = {'success':0, 'fail':0}
        # self.observation_space = None 
        # self.action_space = None 

    def reset(self, train_mode, config):
        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
        self.state_tracker.initialize_episode()
        user_action = self.user.initialize_episode()
        self.print_function(user_action = user_action)
        self.state_tracker.update(user_action = user_action)
        state_vector = self.prepare_state_representation(self.state_tracker.get_state_for_agent())
        self.env_info = [State(state_vector, 0, False)] 
        return self.env_info 

    def step(self, action):
        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        agent_action = self.action_decode(action)
        self.state_tracker.update(agent_action=agent_action)
        self.print_function(agent_action = agent_action['act_slot_response'])
        
        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################
        sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        user_action, session_over, dialog_status = self.user.next(sys_action)
        reward = self.reward_function(dialog_status)
        
        ########################################################################
        #   Update state tracker with latest user action
        ########################################################################
        if session_over != True:
            self.state_tracker.update(user_action = user_action)
            self.print_function(user_action = user_action)
        else:
            if reward > 0:
                self.stat['success'] += 1
            else: self.stat['fail'] += 1

        state_vector = self.prepare_state_representation(self.state_tracker.get_state_for_agent())
        self.env_info = [State(state_vector, reward, session_over)] 

        return self.env_info 

    def reward_function(self, dialog_status):
        """ Reward Function 1: a reward function based on the dialog_status """
        if dialog_status == FAILED_DIALOG:
            reward = -self.user.max_turn #10
        elif dialog_status == SUCCESS_DIALOG:
            reward = 2*self.user.max_turn #20
        else:
            reward = -1
        return reward
    
    def reward_function_without_penalty(self, dialog_status):
        """ Reward Function 2: a reward function without penalty on per turn and failure dialog """
        if dialog_status == FAILED_DIALOG:
            reward = 0
        elif dialog_status == SUCCESS_DIALOG:
            reward = 2*self.user.max_turn
        else:
            reward = 0
        return reward
    
    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """
        
        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
    
    
    def action_decode(self, action):
        """ DQN: Input state, output action """
        if isinstance(action, np.ndarray):
            action = action[0]
        act_slot_response = deepcopy(self.feasible_actions[action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
        
    
    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        
        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']
        
        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep =  np.zeros((1, self.act_cardinality))
        user_act_rep[0,self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1,self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0,self.slot_set[slot]] = 1.0
        
        turn_rep = np.zeros((1,1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        self.final_representation = np.squeeze(np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep]))
        return self.final_representation

    def action_index(self, act_slot_response):
        """ Return the index of action """
        
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print(act_slot_response)
        raise Exception("action index not found")
        return None
  
    def print_function(self, agent_action=None, user_action=None):
        """ Print Function """
            
        if agent_action:
            if run_mode == 0:
                print("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            elif run_mode == 1:
                print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
            elif run_mode == 2: # debug mode
                print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
                print("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            
            if auto_suggest == 1:
                print('(Suggested Values: %s)' % (self.state_tracker.get_suggest_slots_values(agent_action['request_slots'])))
        elif user_action:
            if run_mode == 0:
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            elif run_mode == 1: 
                print ("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
            elif run_mode == 2: # debug mode, show both
                print ("Turn %d usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))

    def rule_policy(self):
        """ Rule Policy """
        
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

    def close(self):
        print('\nstatistics: %s' % (self.stat))
        try:
            print('\nsuccess rate:', (self.stat['success']/(self.stat['success'] + self.stat['fail'])))
        except:
            pass
        print("close")


class MovieEnv(BaseEnv):
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
        super(MovieEnv, self).__init__(spec, e, env_space)
        util.set_attr(self, self.env_spec, [
            'observation_dim',
            'action_dim',
        ])
        worker_id = int(f'{os.getpid()}{self.e+int(ps.unique_id())}'[-4:])
        # TODO dynamically compose components according to env_spec
        self.u_env = MovieActInActOutEnvironment(worker_id)
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
        observation_shape = (self.env_spec.get('observation_dim'),)
        observation_space = spaces.Box(low=0, high=1, shape=observation_shape, dtype=np.int32)
        set_gym_space_attr(observation_space)
        action_space = spaces.Discrete(self.env_spec.get('action_dim'))
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
