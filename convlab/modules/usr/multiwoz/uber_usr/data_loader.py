#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

__time__ = '2019/2/25 16:22'

import collections
import json
import os
import pickle
import random

import numpy as np

__time__ = '2019/2/23 16:41'

# domains_set = {'restaurant', 'attraction', 'hotel', 'hospital', 'police', 'train', 'taxi'}
domains_set = {'restaurant', 'attraction', 'hotel', 'train', 'taxi'}

ref_slot_data2stand = {
    'train': {
        'Duration': 'Time', 'Price': 'Ticket', 'TrainID': 'Id'
    }
}

SysDa2Goal = {
    'attraction': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Fee': "entrance fee", 'Name': "name", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Type': "type",
        'none': None
    },
    'booking': {
        'Day': 'day', 'Name': 'name', 'People': 'people',
        'Ref': 'ref', 'Stay': 'stay', 'Time': 'time',
        'none': None
    },
    'hotel': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Internet': "internet", 'Name': "name", 'Parking': "parking",
        'Phone': "phone", 'Post': "postcode", 'Price': "pricerange",
        'Ref': "ref", 'Stars': "stars", 'Type': "type",
        'none': None
    },
    'restaurant': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Name': "name", 'Food': "food", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Ref': "ref",
        'none': None
    },
    'taxi': {
        'Arrive': "arriveBy", 'Car': "car type", 'Depart': "departure",
        'Dest': "destination", 'Leave': "leaveAt", 'Phone': "phone",
        'none': None
    },
    'train': {
        'Arrive': "arriveBy", 'Choice': "choice", 'Day': "day",
        'Depart': "departure", 'Dest': "destination", 'Id': "trainID", 'TrainID': "trainID",
        'Leave': "leaveAt", 'People': "people", 'Ref': "ref",
        'Ticket': "price", 'Time': "duration", 'Duration': 'duration', 'none': None
    }
}

UsrDa2Goal = {
    'attraction': {
        'Area': 'area', 'Name': 'name', 'Type': 'type',
        'Addr': 'address', 'Fee': 'entrance fee', 'Phone': 'phone',
        'Post': 'postcode', 'none': None
    },
    'hospital': {
        'Department': 'department', 'Addr': 'address', 'Phone': 'phone',
        'Post': 'postcode', 'none': None
    },
    'hotel': {
        'Area': 'area', 'Internet': 'internet', 'Name': 'name',
        'Parking': 'parking', 'Price': 'pricerange', 'Stars': 'stars',
        'Type': 'type', 'Addr': 'address', 'Phone': 'phone',
        'Post': 'postcode', 'Day': 'day', 'People': 'people',
        'Stay': 'stay', 'none': None
    },
    'police': {
        'Addr': 'address', 'Phone': 'phone', 'Post': 'postcode', 'none': None
    },
    'restaurant': {
        'Area': 'area', 'Day': 'day', 'Food': 'food',
        'Name': 'name', 'People': 'people', 'Price': 'pricerange',
        'Time': 'time', 'Addr': 'address', 'Phone': 'phone',
        'Post': 'postcode', 'none': None
    },
    'taxi': {
        'Arrive': 'arriveBy', 'Depart': 'departure', 'Dest': 'destination',
        'Leave': 'leaveAt', 'Car': 'car type', 'Phone': 'phone', 'none': None
    },
    'train': {
        'Time': "duration", 'Arrive': 'arriveBy', 'Day': 'day', 'Ref': "ref",
        'Depart': 'departure', 'Dest': 'destination', 'Leave': 'leaveAt',
        'People': 'people', 'Duration': 'duration', 'Price': 'price', 'Choice': "choice",
        'TrainID': 'trainID', 'Ticket': 'price', 'Id': "trainID", 'none': None
    }
}


class DataLoader(object):

    def __init__(self):
        self.__org_goals = None
        self.__org_usr_dass = None
        self.__org_sys_dass = None

        self.__goals = None
        self.__usr_dass = None
        self.__sys_dass = None

        self.__goals_seg = None
        self.__usr_dass_seg = None
        self.__sys_dass_seg = None

        self.__voc_goal = None
        self.__voc_usr = None
        self.__voc_usr_rev = None
        self.__voc_sys = None

        self.pad = '<PAD>'
        self.unk = '<UNK>'
        self.sos = '<SOS>'
        self.eos = '<EOS>'
        self.special_words = [self.pad, self.unk, self.sos, self.eos]
        pass

    @ staticmethod
    def usrgoal2seq(goal: dict):
        def add(lt: list, domain_goal: dict, domain: str, intent: str):
            map = domain_goal.get(intent, {})
            slots = [slot for slot in map.keys() if isinstance(map[slot], str)]
            if len(slots) > 0:
                lt.append(domain + '_' + intent)
                lt.append('(')
                slots.sort()
                lt.extend(slots)
                lt.append(')')

        ret = []
        for (domain, domain_goal) in filter(lambda x: (x[0] in domains_set and len(x[1]) > 0),
                                            sorted(goal.items(), key=lambda x: x[0])):
            # info
            add(ret, domain_goal, domain, 'info')
            # fail_info
            add(ret, domain_goal, domain, 'fail_info')
            # book
            add(ret, domain_goal, domain, 'book')
            # fail_book
            add(ret, domain_goal, domain, 'fail_book')
            # reqt
            slots = domain_goal.get('reqt', [])
            if len(slots) > 0:
                ret.append(domain + '_' + 'reqt')
                ret.append('(')
                ret.extend(slots)
                ret.append(')')

        return ret

    @staticmethod
    def query_goal_for_sys(domain, slot, value, goal):
        ret = None
        if slot == 'Choice':
            ret = 'Zero' if value in ['None', 'none'] else 'NonZero'
        else:
            goal_slot = SysDa2Goal.get(domain, {}).get(slot, 'Unknow')
            if goal_slot is not None and goal_slot != 'Unknow':
                check = {'info': None, 'fail_info': None, 'book': None, 'fail_book': None}
                for zone in check.keys():
                    dt = goal.get(domain, {}).get(zone, {})
                    if goal_slot in dt.keys():
                        check[zone] = (dt[goal_slot].lower() == value.lower())
                if True in check.values() or False in check.values():
                    # in constraint
                    ret = 'InConstraint'
                    if True in check.values() and False in check.values():
                        ret += '[C]'
                    elif True in check.values() and False not in check.values():
                        ret += '[R]'
                    elif True not in check.values() and False in check.values():
                        ret += '[W]'
                elif goal_slot in goal.get(domain, {}).get('reqt', []):
                    # in Request
                    ret = 'InRequest'
                else:
                    # not in goal
                    ret = 'NotInGoal'
            elif goal_slot == 'Unknow':
                ret = 'Unknow'
            else:
                ret = None
        return ret

    @staticmethod
    def sysda2seq(sys_da: dict, goal: dict):
        ret = []
        for (act, pairs) in sorted(sys_da.items(), key=lambda x: x[0]):
            (domain, intent) = act.split('-')
            domain = domain.lower()
            intent = intent.lower()
            if domain in domains_set or domain in ['booking', 'general']:
                ret.append(act)
                ret.append('(')
                if 'general' not in act:
                    for [slot, val] in sorted(pairs, key=lambda x: x[0]):
                        if (intent == 'request'):
                            ret.append(slot)
                        else:
                            m_val = DataLoader.query_goal_for_sys(domain, slot, val, goal)
                            if m_val is not None:
                                ret.append(slot + '=' + m_val)

                ret.append(')')
        # assert len(ret) > 0, str(sys_da)
        return ret

    @staticmethod
    def query_goal_for_usr(domain, slot, value, goal):
        ret = None
        goal_slot = UsrDa2Goal.get(domain, {}).get(slot, 'Unknow')
        if goal_slot is not None and goal_slot != 'Unknow':
            check = {'info': None, 'fail_info': None, 'book': None, 'fail_book': None}
            for zone in check.keys():
                dt = goal.get(domain, {}).get(zone, {})
                if goal_slot in dt.keys():
                    check[zone] = (dt[goal_slot].replace(' ', '').lower() == value.replace(' ', '').lower())
            if True in check.values() or False in check.values():
                # in constraint
                if True in check.values():
                    ret = 'In' + [zone for (zone, value) in check.items() if value == True][0].capitalize()
                else:
                    ret = 'In' + [zone for (zone, value) in check.items() if value == False][0].capitalize()
            else:
                # not in goal
                ret = 'DoNotCare'
        elif goal_slot == 'Unknow':
            ret = 'Unknow'
        else:
            ret = None
        return ret

    @staticmethod
    def usrda2seq(usr_da: dict, goal: dict):
        ret = []
        for (act, pairs) in sorted(usr_da.items(), key=lambda x: x[0]):
            (domain, intent) = act.split('-')
            domain = domain.lower()
            intent = intent.lower()
            if domain in domains_set or domain in ['booking', 'general']:
                ret.append(act)
                ret.append('(')
                if 'general' not in act:
                    for [slot, val] in sorted(pairs, key=lambda x: x[0]):
                        if (intent == 'request'):
                            ret.append(slot)
                        else:
                            m_val = DataLoader.query_goal_for_usr(domain, slot, val, goal)
                            if m_val is not None:
                                ret.append(slot + '=' + m_val)

                ret.append(')')
        assert len(ret) > 0, str(usr_da)
        return ret

    @staticmethod
    def usrseq2da(usr_seq: list, goal: dict):
        ret = {}
        cur_act = None
        domain = None
        for word in usr_seq:
            if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '(', ')']:
                continue

            # act
            if '-' in word:
                cur_act = word
                domain, _ = cur_act.split('-')
                ret[cur_act] = []

            # slot-value
            elif '=' in word and cur_act is not None:
                slot_da, value_pos = word.split('=')
                value_pos = value_pos.lower()
                slot_goal = UsrDa2Goal.get(domain.lower(), {}).get(slot_da, None)
                if slot_goal is not None:
                    value = None
                    if value_pos == 'ininfo':
                        value = goal.get(domain.lower(), {}).get('info', {}).get(slot_goal, None)
                    elif value_pos == 'infail_info':
                        value = goal.get(domain.lower(), {}).get('fail_info', {}).get(slot_goal, None)
                    elif value_pos == 'inbook':
                        value = goal.get(domain.lower(), {}).get('book', {}).get(slot_goal, None)
                    elif value_pos == 'infail_book':
                        value = goal.get(domain.lower(), {}).get('fail_book', {}).get(slot_goal, None)
                    elif value_pos == 'donotcare':
                        value = 'don\'t care'

                    if value is not None:
                        ret[cur_act].append([slot_da, value])
                    else:
                        pass
                        #assert False, slot_da
                else:
                    pass
                    # assert False, '%s - %s' % (domain, slot_da)

            # slot in reqt
            elif cur_act is not None:
                ret[cur_act].append([word, '?'])

        return ret

    @staticmethod
    def ref_data2stand(da):
        for act in da.keys():
            if '-' in act:
                domain, _ = act.split('-')
                for idx in range(len(da[act])):
                    da[act][idx][0] = ref_slot_data2stand.get(domain.lower(), {}).get(da[act][idx][0], da[act][idx][0])
        return da


    def org_data_loader(self):
        if self.__org_goals is None or self.__org_usr_dass is None or self.__org_sys_dass is None:
            file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                'data/multiwoz/annotated_user_da_with_span_full.json')
            # 'data/multiwoz/annotated_user_da_with_span_100sample.json')
            full_data = json.load(open(file_path))
            goals = []
            usr_dass = []
            sys_dass = []
            for session in full_data.values():
                goal = session.get('goal', {})
                logs = session.get('log', [])
                usr_das, sys_das = [], []
                for turn in range(len(logs) // 2):
                    # <class 'dict'>: {'Hotel-Inform': [['Price', 'cheap'], ['Type', 'hotel']]}
                    usr_da = self.ref_data2stand(logs[turn * 2].get('dialog_act', {}))
                    sys_da = self.ref_data2stand(logs[turn * 2 + 1].get('dialog_act', {}))
                    usr_das.append(usr_da)
                    sys_das.append(sys_da)
                    if len(usr_das[-1]) <= 0 or len(sys_das[-1]) <= 0:
                        usr_das = []
                        break

                if len(goal) <= 0 or len(logs) <= 0 or \
                        len(usr_das) <= 0 or len(sys_das) <= 0 or len(usr_das) != len(sys_das):
                    continue
                goals.append(goal)
                usr_dass.append(usr_das)
                sys_dass.append(sys_das)

            self.__org_goals = [DataLoader.usrgoal2seq(goal) for goal in goals]
            self.__org_usr_dass = [[DataLoader.usrda2seq(usr_da, goal) for usr_da in usr_das] for (usr_das, goal) in zip(usr_dass, goals)]
            self.__org_sys_dass = [[DataLoader.sysda2seq(sys_da, goal) for sys_da in sys_das] for (sys_das, goal) in zip(sys_dass, goals)]

        return self.__org_goals, self.__org_usr_dass, self.__org_sys_dass

    def vocab_loader(self):
        if self.__voc_goal is None or self.__voc_usr is None or self.__voc_usr_rev is None or self.__voc_sys is None:
            vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.pkl')

            if os.path.exists(vocab_path):
                self.__voc_goal, self.__voc_usr, self.__voc_sys = pickle.load(open(vocab_path, 'rb'))
            else:
                goals, usr_dass, sys_dass = self.org_data_loader()

                # goals
                counter = collections.Counter()
                for goal in goals:
                    for word in goal:
                        counter[word] += 1
                word_list = self.special_words + [x for x in counter.keys()]
                self.__voc_goal = {x: i for i, x in enumerate(word_list)}

                counter = collections.Counter()
                for usr_das in usr_dass:
                    for usr_da in usr_das:
                        for word in usr_da:
                            counter[word] += 1
                word_list = self.special_words + [x for x in counter.keys()]
                self.__voc_usr = {x: i for i, x in enumerate(word_list)}

                counter = collections.Counter()
                for sys_das in sys_dass:
                    for sys_da in sys_das:
                        for word in sys_da:
                            counter[word] += 1
                word_list = self.special_words + [x for x in counter.keys()]
                self.__voc_sys = {x: i for i, x in enumerate(word_list)}

                pickle.dump((self.__voc_goal, self.__voc_usr, self.__voc_sys), open(vocab_path, 'wb'))
                print('voc build ok')

            self.__voc_usr_rev = {val: key for (key, val) in self.__voc_usr.items()}

        return self.__voc_goal, self.__voc_usr, self.__voc_sys

    def get_voc_size(self):
        goal, usr, sys = self.vocab_loader()
        return len(goal), len(usr), len(sys)

    def data_loader(self):
        if self.__goals is None or self.__usr_dass is None or self.__sys_dass is None:
            org_goals, org_usr_dass, org_sys_dass = self.org_data_loader()
            voc_goal, voc_usr, voc_sys = self.vocab_loader()
            self.__goals = [[voc_goal.get(word, voc_goal[self.unk]) for word in goal] for goal in org_goals]
            self.__usr_dass = [
                [[voc_usr[self.sos]] + [voc_usr.get(word, voc_usr[self.unk]) for word in usr_da] + [voc_usr[self.eos]]
                 for usr_da in usr_das] for usr_das in org_usr_dass]
            self.__sys_dass = [[[voc_sys.get(word, voc_sys[self.unk]) for word in sys_da] for sys_da in sys_das] for
                               sys_das in org_sys_dass]
        assert len(self.__goals) == len(self.__usr_dass)
        assert len(self.__goals) == len(self.__sys_dass)
        return self.__goals, self.__usr_dass, self.__sys_dass

    @staticmethod
    def train_test_val_split(goals, usr_dass, sys_dass, test_size=0.1, val_size=0.1):
        idx = range(len(goals))
        idx_test = random.sample(idx, int(len(goals) * test_size))
        idx_train = list(set(idx) - set(idx_test))
        idx_val = random.sample(idx_train, int(len(goals) * val_size))
        idx_train = list(set(idx_train) - set(idx_val))
        idx_train = random.sample(idx_train, len(idx_train))
        return np.array(goals)[idx_train], np.array(usr_dass)[idx_train], np.array(sys_dass)[idx_train], \
               np.array(goals)[idx_test], np.array(usr_dass)[idx_test], np.array(sys_dass)[idx_test], \
               np.array(goals)[idx_val], np.array(usr_dass)[idx_val], np.array(sys_dass)[idx_val]

    def data_loader_seg(self):
        if self.__goals_seg is None or self.__usr_dass_seg is None or self.__sys_dass_seg is None:
            self.data_loader()
            self.__goals_seg, self.__usr_dass_seg, self.__sys_dass_seg = [], [], []

            for (goal, usr_das, sys_das) in zip(self.__goals, self.__usr_dass, self.__sys_dass):
                goals, usr_dass, sys_dass = [], [], []
                for length in range(len(usr_das)):
                    goals.append(goal)
                    usr_dass.append([usr_das[idx] for idx in range(length + 1)])
                    sys_dass.append([sys_das[idx] for idx in range(length + 1)])

                self.__goals_seg.append(goals)
                self.__usr_dass_seg.append(usr_dass)
                self.__sys_dass_seg.append(sys_dass)

        assert len(self.__goals_seg) == len(self.__usr_dass_seg)
        assert len(self.__goals_seg) == len(self.__sys_dass_seg)
        return self.__goals_seg, self.__usr_dass_seg, self.__sys_dass_seg

    def id2sentence(self, ids):
        sentence = [self.__voc_usr_rev[id] for id in ids]
        return sentence

    @staticmethod
    def train_test_val_split_seg(goals_seg, usr_dass_seg, sys_dass_seg, test_size=0.1, val_size=0.1):
        def dr(dss):
            return np.array([d for ds in dss for d in ds])

        idx = range(len(goals_seg))
        idx_test = random.sample(idx, int(len(goals_seg) * test_size))
        idx_train = list(set(idx) - set(idx_test))
        idx_val = random.sample(idx_train, int(len(goals_seg) * val_size))
        idx_train = list(set(idx_train) - set(idx_val))
        idx_train = random.sample(idx_train, len(idx_train))
        return dr(np.array(goals_seg)[idx_train]), dr(np.array(usr_dass_seg)[idx_train]), dr(np.array(sys_dass_seg)[idx_train]), \
               dr(np.array(goals_seg)[idx_test]), dr(np.array(usr_dass_seg)[idx_test]), dr(np.array(sys_dass_seg)[idx_test]), \
               dr(np.array(goals_seg)[idx_val]), dr(np.array(usr_dass_seg)[idx_val]), dr(np.array(sys_dass_seg)[idx_val])


def batch_iter(x, y, z, batch_size=64):
    data_len = len(x)
    num_batch = ((data_len - 1) // batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = np.array(x)[indices]
    y_shuffle = np.array(y)[indices]
    z_shuffle = np.array(z)[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], z_shuffle[start_id:end_id]



def test():
    # example
    # data class
    data = DataLoader()
    seq_goals, seq_usr_dass, seq_sys_dass = data.data_loader_seg()

    # DO NOT DELETE
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'data/multiwoz/annotated_user_da_with_span_full.json')
    # 'data/multiwoz/annotated_user_da_with_span_100sample.json')
    full_data = json.load(open(file_path))
    dic_goals = []
    for session in full_data.values():
        goal = session.get('goal', {})
        logs = session.get('log', [])
        usr_das, sys_das = [], []
        for turn in range(len(logs) // 2):
            usr_das.append(logs[turn * 2].get('dialog_act', {}))
            sys_das.append(logs[turn * 2 + 1].get('dialog_act', {}))
            if len(usr_das[-1]) <= 0 or len(sys_das[-1]) <= 0:
                usr_das = []
                break

        if len(goal) <= 0 or len(logs) <= 0 or \
                len(usr_das) <= 0 or len(sys_das) <= 0 or len(usr_das) != len(sys_das):
            continue
        dic_goals.append(goal)

    
    _, org_usr_dass, org_sys_dass = data.org_data_loader()
    for goal, usr_das, sys_das in zip(dic_goals, org_usr_dass, org_sys_dass):
        # in session
        for usr_da_seq, _ in zip(usr_das, sys_das):
            usr_da = DataLoader.usrseq2da(usr_da_seq, goal)
            print(usr_da_seq)
            print('-> ', usr_da)
            pass
    print('')
    '''
    # segment
    seq_goals, seq_usr_dass, seq_sys_dass = data.data_loader_seg()
    # split train and test
    train_goals, train_usrdas, train_sysdas, test_goals, test_usrdas, test_sysdas, val_goals, val_usrdas, val_sysdas = DataLoader.train_test_val_split_seg(
        seq_goals, seq_usr_dass, seq_sys_dass)

    # batch generator
    generator = batch_iter(train_goals, train_usrdas, train_sysdas, 10)

    # show
    for batch_goals, batch_usrdas, batch_sysdas in generator:
        print('----------------------------------------')
        print(batch_goals)
        print(batch_usrdas)
        print(batch_sysdas)
    '''

if __name__ == '__main__':
    test()