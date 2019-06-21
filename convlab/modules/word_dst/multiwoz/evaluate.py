# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from convlab.modules.dst.multiwoz.dst_util import minDistance
from convlab.modules.word_dst.multiwoz.mdbt import MDBTTracker


class Word_DST:
    """A temporary semi-finishingv agent for word_dst testing, which takes as input utterances and output dialog state."""
    def __init__(self):
        self.dst = MDBTTracker(data_dir='../../../../data/mdbt')
        self.nlu = None

    def update(self, action, observation):
        # update history
        self.dst.state['history'].append([str(action)])

        # NLU parsing
        input_act = self.nlu.parse(observation, sum(self.dst.state['history'], [])) if self.nlu else observation

        # state tracking
        self.dst.update(input_act)
        self.dst.state['history'][-1].append(observation)

        # update history
        return self.dst.state

    def reset(self):
        self.dst.init_session()

def load_data(path='../../../../data/multiwoz/test.json'):
    """Load data (mainly for testing data)."""
    data = json.load(open(path))
    result = []
    for id, session in data.items():
        log = session['log']
        turn_data = ['null']
        goal = session['goal']
        session_data = []
        for turn_idx, turn in enumerate(log):
            if turn_idx % 2 == 0:  # user
                observation = turn['text']
                turn_data.append(observation)
            else:  # system
                action = turn['text']
                golden_state = turn['metadata']
                turn_data.append(golden_state)
                session_data.append(turn_data)
                turn_data = [action]
        result.append([session_data, goal])
    return result

def run_test():
    agent = Word_DST()
    agent.reset()

    test_data = load_data()
    test_result = []
    for session_data, goal in test_data:
        session_result = []
        for action, observation, golden_state in session_data:
            pred_state = agent.update(action, observation)
            session_result.append([golden_state, pred_state['belief_state']])
        test_result.append([session_result, goal])
        agent.reset()
    return test_result

class ResultStat:
    """A functional class for accuracy statistic."""
    def __init__(self):
        self.stat = {}

    def add(self, domain, slot, score):
        """score = 1 or 0"""
        if domain in self.stat:
            if slot in self.stat[domain]:
                self.stat[domain][slot][0] += score
                self.stat[domain][slot][1] += 1.
            else:
                self.stat[domain][slot] = [score, 1.]
        else:
            self.stat[domain] = {slot: [score, 1.]}

    def domain_acc(self, domain):
        domain_stat = self.stat[domain]
        ret = [0, 0]
        for _, r in domain_stat.items():
            ret[0] += r[0]
            ret[1] += r[1]
        return ret[0]/(ret[1] + 1e-10)

    def slot_acc(self, domain, slot):
        slot_stat = self.stat[domain][slot]
        return slot_stat[0] / (slot_stat[1] + 1e-10)

    def all_acc(self):
        acc_result = {}
        for domain in self.stat:
            acc_result[domain] = {}
            acc_result[domain]['acc'] = self.domain_acc(domain)
            for slot in self.stat[domain]:
                acc_result[domain][slot+'_acc'] = self.slot_acc(domain, slot)
        return json.dumps(acc_result, indent=4)

def evaluate(test_result):
    stat = ResultStat()
    session_level = [0., 0.]
    for session, goal in test_result:
        last_pred_state = None
        for golden_state, pred_state in session:  # session
            last_pred_state = pred_state
            domains = golden_state.keys()
            for domain in domains:   # domain
                if domain == 'bus':
                    continue
                assert domain in pred_state, 'domain: {}'.format(domain)
                golden_domain, pred_domain = golden_state[domain], pred_state[domain]
                for slot, value in golden_domain['semi'].items():  # slot
                    if _is_empty(slot, golden_domain['semi']):
                        continue
                    pv = pred_domain['semi'][slot] if slot in pred_domain['semi'] else '_None'
                    score = 0.
                    if _is_match(value, pv):
                        score = 1.
                    stat.add(domain, slot, score)
        if match_goal(last_pred_state, goal):
            session_level[0] += 1
        session_level[1] += 1
    print('domain and slot-level acc:')
    print(stat.all_acc())
    print('session-level acc: {}'.format(convert2acc(session_level[0], session_level[1])))

def convert2acc(a, b):
    if b == 0:
        return -1
    return a/b

def match_goal(pred_state, goal):
    domains = pred_state.keys()
    for domain in domains:
        if domain not in goal:
            continue
        goal_domain = goal[domain]
        if 'info' not in goal_domain:
            continue
        goal_domain_info = goal_domain['info']
        for slot, value in goal_domain_info.items():
            if slot in pred_state[domain]['semi']:
                v = pred_state[domain]['semi'][slot]
            else:
                return False
            if _is_match(value, v):
                continue
            elif _fuzzy_match(value, v):
                continue
            else:
                return False
    return True

def _is_empty(slot, domain_state):
    if slot not in domain_state:
        return True
    value = domain_state[slot]
    if value is None or value == "" or value == 'null':
        return True
    return False

def _is_match(value1, value2):
    if not isinstance(value1, str) or not isinstance(value2, str):
        return value1 == value2
    value1 = value1.lower()
    value2 = value2.lower()
    value1 = ' '.join(value1.strip().split())
    value2 = ' '.join(value2.strip().split())
    if value1 == value2:
        return True
    return False

def _fuzzy_match(value1, value2):
    if not isinstance(value1, str) or not isinstance(value2, str):
        return value1 == value2
    value1 = value1.lower()
    value2 = value2.lower()
    value1 = ' '.join(value1.strip().split())
    value2 = ' '.join(value2.strip().split())
    d = minDistance(value1, value2)
    if (len(value1) >= 10 and d <= 2) or (len(value1) >= 15 and d <= 3):
        return True
    return False

if __name__ == '__main__':
    test_result = run_test()
    json.dump(test_result, open('word_dst_test_result.json', 'w+'), indent=2)
    print('test session num: {}'.format(len(test_result)))
    evaluate(test_result)