# -*- coding: utf-8 -*-

import numpy as np
import random

from convlab.evaluator.evaluator import Evaluator 
from convlab.modules.util.dbquery import query

requestable = \
{'attraction': ['post', 'phone', 'addr', 'fee', 'area', 'type'],
 'restaurant': ['addr', 'phone', 'post', 'ref', 'price', 'area', 'food'],
 'train': ['ticket', 'time', 'ref', 'id', 'arrive', 'leave'],
 'hotel': ['addr', 'post', 'phone', 'ref', 'price', 'internet', 'parking', 'area', 'type', 'stars'],
 'taxi': ['car', 'phone'],
 'hospital': ['post', 'phone', 'addr'],
 'police': ['addr', 'post', 'phone']}

belief_domains = requestable.keys()

mapping = {'restaurant': {'addr': 'address', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'price': 'pricerange'},
        'hotel': {'addr': 'address', 'area': 'area', 'internet': 'internet', 'parking': 'parking', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'price': 'pricerange', 'stars': 'stars', 'type': 'type'},
        'attraction': {'addr': 'address', 'area': 'area', 'fee': 'entrance fee', 'name': 'name', 'phone': 'phone', 'post': 'postcode', 'type': 'type'},
        'train': {'id': 'trainID', 'arrive': 'arriveBy', 'day': 'day', 'depart': 'departure', 'dest': 'destination', 'time': 'duration', 'leave': 'leaveAt', 'ticket': 'price'},
        'taxi': {'car': 'car type', 'phone': 'phone'},
        'hospital': {'post': 'postcode', 'phone': 'phone', 'addr': 'address', 'department': 'department'},
        'police': {'post': 'postcode', 'phone': 'phone', 'addr': 'address'}}

class MultiWozEvaluator(Evaluator):
    def __init__(self):
        self.sys_da_array = []
        self.usr_da_array = []
        self.state_array = []
        self.goal = {}
        self.cur_domain = ''
        self.booked = {}

    def _init_dict(self):
        dic = {}
        for domain in belief_domains:
            dic[domain] = {'info':{}, 'book':{}, 'reqt':[]}
        return dic
    
    def _init_dict_booked(self):
        dic = {}
        for domain in belief_domains:
            dic[domain] = None
        return dic
    
    def _expand(self, goal):
        for domain in belief_domains:
            if domain not in goal:
                goal[domain] = {'info':{}, 'book':{}, 'reqt':[]}
                continue
            if 'info' not in goal[domain]:
                goal[domain]['info'] = {}
            if 'book' not in goal[domain]:
                goal[domain]['book'] = {}
            if 'reqt' not in goal[domain]:
                goal[domain]['reqt'] = []
        return goal
    
    def add_goal(self, goal):
        """
        init goal and array
        args:
            goal: dict[domain] dict['info'/'book'/'reqt'] dict/dict/list[slot]
        """
        self.sys_da_array = []
        self.usr_da_array = []
        self.state_array = []
        self.goal = goal
        self.cur_domain = ''
        self.booked = self._init_dict_booked()
    
    def add_sys_da(self, da_turn):
        """
        add sys_da into array
        args:
            da_turn: dict[domain-intent] list[slot, value]
        """
        for dom_int in da_turn:
            domain = dom_int.split('-')[0].lower()
            if domain in belief_domains and domain != self.cur_domain:
                self.cur_domain = domain                
            slot_pair = da_turn[dom_int]
            for slot, value in slot_pair:
                da = (dom_int +'-'+slot).lower()
                self.sys_da_array.append(da+'-'+value)
                
                if da == 'booking-book-ref' and self.cur_domain in ['hotel', 'restaurant', 'train']:
                    entities = query(self.cur_domain, self.state_array[-1][self.cur_domain]['semi'].items())
                    if entities and not self.booked[self.cur_domain]:
                        self.booked[self.cur_domain] = random.choice(entities)
                elif da == 'train-offerbook-ref' or da == 'train-inform-ref':
                    entities = query('train', self.state_array[-1]['train']['semi'].items())
                    if entities and not self.booked['train']:
                        self.booked['train'] = random.choice(entities)
                elif da == 'taxi-inform-car':
                    if not self.booked['taxi']:
                        self.booked['taxi'] = 'booked'
                
    def add_usr_da(self, da_turn):
        """
        add usr_da into array
        args:
            da_turn: dict[domain-intent] list[slot, value]
        """
        for dom_int in da_turn:
            slot_pair = da_turn[dom_int]
            for slot, value in slot_pair:
                da = (dom_int +'-'+slot).lower()
                self.usr_da_array.append(da+'-'+value)
        
    def add_state(self, state_turn):
        """
        add state into array
        args:
            state_turn: dict[domain] dict['book'/'semi'] dict[slot]
        """
        self.state_array.append(state_turn['belief_state'])

    def _book_rate_goal(self, goal, booked_entity):
        """
        judge if the selected entity meets the constraint
        """
        score = []
        for domain in belief_domains:
            if goal[domain]['book']:
                tot = len(goal[domain]['info'].keys())
                if tot == 0:
                    continue
                entity = booked_entity[domain]
                if entity is None:
                    score.append(0)
                    continue
                if domain == 'taxi':
                    score.append(1)
                    continue
                match = 0
                for k, v in goal[domain]['info'].items():
                    if k in ['destination', 'departure', 'name']:
                        tot -= 1
                    elif k == 'leaveAt':
                        try:
                            v_constraint = int(v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['leaveAt'].split(':')[0]) * 100 + int(entity['leaveAt'].split(':')[1])
                            if v_constraint <= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    elif k == 'arriveBy':
                        try:
                            v_constraint = int(v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['arriveBy'].split(':')[0]) * 100 + int(entity['arriveBy'].split(':')[1])
                            if v_constraint >= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    else:
                        if v.strip() == entity[k].strip():
                            match += 1
                if tot != 0:
                    score.append(match / tot)
        return score
    
    def _inform_F1_goal(self, goal, sys_history):
        """
        judge if all the requested information is answered
        """
        inform_slot = {}
        for domain in belief_domains:
            inform_slot[domain] = set()
        for da in sys_history:
            domain, intent, slot, value = da.split('-', 3)
            if intent in ['inform', 'recommend', 'offerbook', 'offerbooked'] and domain in belief_domains and slot in mapping[domain]:
                inform_slot[domain].add(mapping[domain][slot])
        TP, FP, FN = 0, 0, 0
        for domain in belief_domains:
            for k in goal[domain]['reqt']:
                if k in inform_slot[domain]:
                    TP += 1
                else:
                    FN += 1
                    if domain == 'hotel':
                        pass
            for k in inform_slot[domain]:
                # exclude slots that are informed by users
                if k not in goal[domain]['reqt'] \
                and k not in goal[domain]['info'] \
                and k in requestable[domain]:
                    FP += 1
        return TP, FP, FN
    
    def book_rate(self, ref2goal=True, aggregate=True):
        if ref2goal:
            goal = self._expand(self.goal)
        else:
            goal = self._init_dict()
            for domain in belief_domains:
                goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if i == 'inform' and s in mapping[d]:
                    goal[d]['info'][s] = v
        score = self._book_rate_goal(goal, self.booked)
        if aggregate:
            return np.mean(score) if score else None
        else:
            return score

    def inform_F1(self, ref2goal=True, aggregate=True):
        if ref2goal:
            goal = self._expand(self.goal)
        else:
            goal = self._init_dict()
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][s] = v
                elif i == 'request':
                    goal[d]['reqt'].append(s)
        TP, FP, FN = self._inform_F1_goal(goal, self.sys_da_array)
        if aggregate:
            try:
                rec = TP / (TP + FN)
            except ZeroDivisionError:
                return None, None, None
            try:
                prec = TP / (TP + FP)
                F1 = 2 * prec * rec / (prec + rec)
            except ZeroDivisionError:
                return 0, rec, 0
            return prec, rec, F1
        else:    
            return [TP, FP, FN]
        
    def task_success(self, ref2goal=True):
        book_sess = self.book_rate(ref2goal)
        inform_sess = self.inform_F1(ref2goal)
        # book rate == 1 & inform recall == 1
        # if (book_sess == 1 and inform_sess[1] == 1) \
        # or (book_sess == 1 and inform_sess[1] is None) \
        # or (book_sess is None and inform_sess[1] == 1):
        #     return 1
        if (inform_sess[1] == 1) \
        or (inform_sess[1] is None):
            return 1
        else:
            return 0
