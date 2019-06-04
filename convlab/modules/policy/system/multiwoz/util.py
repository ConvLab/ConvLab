"""
Utility package for system policy
"""

import os
import json
from pprint import pprint
import numpy as np

from convlab.modules.policy.system.multiwoz.rule_based_multiwoz_bot import generate_car, generate_ref_num
from convlab.modules.util.dbquery import query
from convlab.modules.util.multiwoz_slot_trans import REF_USR_DA


DEFAULT_VOCAB_FILE=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), 
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


# action_vocab = ActionVocab()

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


def state_encoder(state):
    db_vector = get_db_state(state['belief_state'])
    book_vector = get_book_state(state['belief_state'])
    info_vector = get_info_state(state['belief_state'])
    request_vector = get_request_state(state['request_state'])
    user_act_vector = get_user_act_state(state['user_action'])
    history_vector = get_history_state(state['history'])

    return np.concatenate((db_vector, book_vector, info_vector, request_vector, user_act_vector, history_vector))


def get_history_state(history):
    history_vector = []

    user_act = None
    repeat_count = 0
    user_act_repeat_vector = [0] * 5
    for turn in reversed(history):
        if user_act == None:
            user_act = turn[1]
        elif user_act == turn[1]:
            repeat_count += 1
        else:
            break
    user_act_repeat_vector[min(4, repeat_count)] = 1
    history_vector += user_act_repeat_vector

    return history_vector


def get_user_act_state(user_act):
    user_act_vector = []

    for domain in REF_SYS_DA:
        for slot in REF_SYS_DA[domain]:
            for act_type in ['Inform', 'Request', 'Booking']:
                domain_act = domain + '-' + act_type
                if domain_act in user_act and slot in [sv[0] for sv in user_act[domain_act]]: 
                    user_act_vector.append(1)
                    # print(domain, act_type, slot)
                else:
                    user_act_vector.append(0)

    return np.array(user_act_vector)


def get_request_state(request_state):
    # TODO fix RuleDST to delete informed slot
    domains = ['taxi', 'restaurant', 'hospital', 'hotel', 'attraction', 'train', 'police']
    request_vector = []

    for domain in domains:
        domain_vector = [0] * (len(REF_USR_DA[domain.capitalize()]) + 1)
        if domain in request_state:
            for slot in request_state[domain]:
                if slot == 'ref':
                    domain_vector[-1] = 1
                else:
                    domain_vector[list(REF_USR_DA[domain.capitalize()].keys()).index(slot)] = 1
                    # print("request:", slot)
        request_vector.extend(domain_vector)

    return np.array(request_vector)


def get_info_state(belief_state):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = ['taxi', 'restaurant', 'hospital', 'hotel', 'attraction', 'train', 'police']
    info_vector = []

    for domain in domains:
        domain_active = False

        booking = []
        for slot in sorted(belief_state[domain]['book'].keys()):
            if slot == 'booked':
                if belief_state[domain]['book']['booked']:
                    booking.append(1)
                else:
                    booking.append(0)
            else:
                if belief_state[domain]['book'][slot] != "":
                    booking.append(1)
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in belief_state[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in belief_state[domain]['book'].keys():
                booking.append(0)
        info_vector += booking

        for slot in belief_state[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if belief_state[domain]['semi'][slot] in ['', 'not mentioned']:
                slot_enc[0] = 1
            elif belief_state[domain]['semi'][slot] == 'dont care' or belief_state[domain]['semi'][slot] == 'dontcare' or belief_state[domain]['semi'][slot] == "don't care":
                slot_enc[1] = 1
                domain_active = True
                # print("dontcare:", slot)
            elif belief_state[domain]['semi'][slot]:
                slot_enc[2] = 1
                domain_active = True
                # print("filled:", slot)
            info_vector += slot_enc

        # quasi domain-tracker
        if domain_active:
            info_vector += [1]
        else:
            info_vector += [0]

    assert len(info_vector) == 94
    return np.array(info_vector)


def get_book_state(belief_state):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if "book" in belief_state['restaurant']:
        if "booked" in belief_state['restaurant']['book']:
            if belief_state['restaurant']['book']["booked"]:
                if "reference" in belief_state['restaurant']['book']["booked"][0]:
                    rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if "book" in belief_state['hotel']:
        if "booked" in belief_state['hotel']['book']:
            if belief_state['hotel']['book']["booked"]:
                if "reference" in belief_state['hotel']['book']["booked"][0]:
                    hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if "book" in belief_state['train']:
        if "booked" in  belief_state['train']['book']:
            if belief_state['train']['book']["booked"]:
                if "reference" in belief_state['train']['book']["booked"][0]:
                    train_vec = np.array([0, 1])

    return np.concatenate((rest_vec, hotel_vec, train_vec)) 


def get_db_state(belief_state):
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    db_vector = np.zeros(6 * len(domains))
    num_entities = {} 
    for domain in domains:
        entities = query(domain, belief_state[domain]['semi'].items())
        db_vector = one_hot(len(entities), domain, domains, db_vector)

    return db_vector 


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