"""
Utility package for system policy
"""

import json
import os

import numpy as np

from convlab.modules.policy.system.multiwoz.rule_based_multiwoz_bot import generate_car, generate_ref_num
from convlab.modules.util.multiwoz.dbquery import query
from convlab.modules.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA

DEFAULT_VOCAB_FILE=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))), 
    "data/multiwoz/da_slot_cnt.json")


class SkipException(Exception):
    def __init__(self):
        pass


class ActionVocab(object):
    def __init__(self, vocab_path=DEFAULT_VOCAB_FILE, num_actions=500):
        # add general actions
        self.vocab = [
            {'general-welcome': ['none']},
            {'general-greet': ['none']},
            {'general-bye': ['none']},
            {'general-reqmore': ['none']}
        ] 
        # add single slot actions
        for domain in REF_SYS_DA:
            for slot in REF_SYS_DA[domain]:
                self.vocab.append({domain + '-Inform': [slot]})
                self.vocab.append({domain + '-Request': [slot]})
        # add actions from stats
        with open(vocab_path, 'r') as f:
            stats = json.load(f)
            for action_string in stats:
                try:
                    act_strings = action_string.split(';];')
                    action_dict = {}
                    for act_string in act_strings:
                        if act_string == '':
                            continue
                        domain_act, slots = act_string.split('[', 1)
                        domain, act_type = domain_act.split('-')
                        if act_type in ['NoOffer', 'OfferBook']:
                            action_dict[domain_act] = ['none'] 
                        elif act_type in ['Select']:
                            if slots.startswith('none'):
                                raise SkipException
                            action_dict[domain_act] = [slots.split(';')[0]] 
                        else:
                            action_dict[domain_act] = sorted(slots.split(';'))
                    if action_dict not in self.vocab:
                        self.vocab.append(action_dict)
                    # else:
                    #     print("Duplicate action", str(action_dict))
                except SkipException as e:
                    print(act_strings)
                if len(self.vocab) >= num_actions:
                    break
        print("{} actions are added to vocab".format(len(self.vocab)))
        # pprint(self.vocab)

    def get_action(self, action_index):
        return self.vocab[action_index]


def _domain_fill(delex_action, state, action, act):
    domain, act_type = act.split('-')
    constraints = []
    for slot in state['belief_state'][domain.lower()]['semi']:
        if state['belief_state'][domain.lower()]['semi'][slot] != "":
            constraints.append([slot, state['belief_state'][domain.lower()]['semi'][slot]])
    if act_type in ['NoOffer', 'OfferBook']:  # NoOffer['none'], OfferBook['none']
        action[act] = []
        for slot in constraints:
            action[act].append([REF_USR_DA[domain].get(slot[0], slot[0]), slot[1]])
    elif act_type in ['Inform', 'Recommend', 'OfferBooked']:  # Inform[Slot,...], Recommend[Slot, ...]
        kb_result = query(domain.lower(), constraints)
        # print("Policy Util")
        # print(constraints)
        # print(len(kb_result))
        if len(kb_result) == 0:
            action[act] = [['none', 'none']]
        else:
            action[act] = []
            for slot in delex_action[act]:
                if slot == 'Choice':
                    action[act].append([slot, len(kb_result)])
                elif slot == 'Ref':
                    action[act].append(["Ref", generate_ref_num(8)])
                else:
                    try:
                        action[act].append([slot, kb_result[0][REF_SYS_DA[domain].get(slot, slot)]])
                    except:
                        action[act].append([slot, "N/A"])
            if len(action[act]) == 0:
                action[act] = [['none', 'none']]
    elif act_type in ['Select']:  # Select[Slot]
        kb_result = query(domain.lower(), constraints)
        if len(kb_result) < 2:
            action[act] = [['none', 'none']]
        else:
            slot = delex_action[act][0]
            action[act] = []
            action[act].append([slot, kb_result[0][REF_SYS_DA[domain].get(slot, slot)]])
            action[act].append([slot, kb_result[1][REF_SYS_DA[domain].get(slot, slot)]])
    else:
        print('Cannot decode:', str(delex_action))
        action[act] = [['none', 'none']]


def action_decoder(state, action_index, action_vocab):
    domains = ['Attraction', 'Hospital', 'Hotel', 'Restaurant', 'Taxi', 'Train', 'Police']
    delex_action = action_vocab.get_action(action_index)
    action = {}

    for act in delex_action:
        domain, act_type = act.split('-')
        if act_type == 'Request':
            action[act] = []
            for slot in delex_action[act]:
                action[act].append([slot, '?'])
        elif act == 'Booking-Book':
            action['Booking-Book'] = [["Ref", generate_ref_num(8)]]
        elif domain not in domains:
            action[act] = [['none', 'none']]
        else:
            if act == 'Taxi-Inform':
                for info_slot in ['leaveAt', 'arriveBy']:
                    if info_slot in state['belief_state']['taxi']['semi'] and \
                        state['belief_state']['taxi']['semi'][info_slot] != "":
                        car = generate_car()
                        phone_num = generate_ref_num(11)
                        action[act] = []
                        action[act].append(['Car', car])
                        action[act].append(['Phone', phone_num])
                        break
                else:
                    action[act] = [['none', 'none']]
            elif act in ['Train-Inform', 'Train-NoOffer', 'Train-OfferBook']:
                for info_slot in ['departure', 'destination']:
                    if info_slot not in state['belief_state']['train']['semi'] or \
                        state['belief_state']['train']['semi'][info_slot] == "":
                        action[act] = [['none', 'none']]
                        break
                else:
                    for info_slot in ['leaveAt', 'arriveBy']:
                        if info_slot in state['belief_state']['train']['semi'] and \
                            state['belief_state']['train']['semi'][info_slot] != "":
                            _domain_fill(delex_action, state, action, act)
                            break
                    else:
                        action[act] = [['none', 'none']]
            elif domain in domains:
                _domain_fill(delex_action, state, action, act)
                
    return action

def one_hot(num, domain, domains, vector):
    """Return number of available entities for particular domain."""
    number_of_options = 6
    if domain != 'train':
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0,0])
        elif num == 1:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector


if __name__ == '__main__':
    pass
