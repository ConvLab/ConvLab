# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from convlab.modules.util.multiwoz.dbquery import query
from convlab.modules.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA


class MultiWozStateEncoder(object):
    def __init__(self):
        pass

    def encode(self, state):
        db_vector = self.get_db_state(state['belief_state'])
        book_vector = self.get_book_state(state['belief_state'])
        info_vector = self.get_info_state(state['belief_state'])
        request_vector = self.get_request_state(state['request_state'])
        user_act_vector = self.get_user_act_state(state['user_action'])
        history_vector = self.get_history_state(state['history'])

        return np.concatenate((db_vector, book_vector, info_vector, request_vector, user_act_vector, history_vector))

    def get_history_state(self, history):
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

    def get_user_act_state(self, user_act):
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

    def get_request_state(self, request_state):
        domains = ['taxi', 'restaurant', 'hospital', 'hotel', 'attraction', 'train', 'police']
        request_vector = []

        for domain in domains:
            domain_vector = [0] * (len(REF_USR_DA[domain.capitalize()]) + 1)
            if domain in request_state:
                for slot in request_state[domain]:
                    if slot == 'ref':
                        domain_vector[-1] = 1
                    else:
                        # print("request: {} {}".format(domain.capitalize(), slot))
                        domain_vector[list(REF_USR_DA[domain.capitalize()].keys()).index(slot)] = 1
                        # print("request:", slot)
            request_vector.extend(domain_vector)

        return np.array(request_vector)

    def get_info_state(self, belief_state):
        """Based on the mturk annotations we form multi-domain belief state"""
        domains = ['taxi', 'restaurant', 'hospital', 'hotel', 'attraction', 'train', 'police']
        info_vector = []

        for domain in domains:
            domain_active = False

            booking = []
            for slot in sorted(belief_state[domain]['book'].keys()):
                if slot == 'booked':
                    if belief_state[domain]['book']['booked'] != []:
                        booking.append(1)
                    else:
                        booking.append(0)
                else:
                    if belief_state[domain]['book'][slot] != "":
                        booking.append(1)
                    else:
                        booking.append(0)
            info_vector += booking

            for slot in belief_state[domain]['semi']:
                slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
                if belief_state[domain]['semi'][slot] in ['', 'not mentioned']:
                    slot_enc[0] = 1
                elif belief_state[domain]['semi'][slot] == 'dont care' or belief_state[domain]['semi'][slot] == 'dontcare' or belief_state[domain]['semi'][slot] == "don't care":
                    slot_enc[1] = 1
                    domain_active = True
                elif belief_state[domain]['semi'][slot]:
                    slot_enc[2] = 1
                    domain_active = True
                info_vector += slot_enc

            # quasi domain-tracker
            if domain_active:
                info_vector += [1]
            else:
                info_vector += [0]

        assert len(info_vector) == 93, f'info_vector {len(info_vector)}'
        return np.array(info_vector)

    def get_book_state(self, belief_state):
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

    def get_db_state(self, belief_state):
        domains = ['restaurant', 'hotel', 'attraction', 'train']
        db_vector = np.zeros(6 * len(domains))
        num_entities = {} 
        for domain in domains:
            entities = query(domain, belief_state[domain]['semi'].items())
            db_vector = self.one_hot(len(entities), domain, domains, db_vector)

        return db_vector 

    def one_hot(self, num, domain, domains, vector):
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


