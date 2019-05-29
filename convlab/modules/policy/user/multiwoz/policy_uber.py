from convlab.modules.usr.multiwoz.goal_generator import GoalGenerator
from convlab.modules.policy.user.policy import UserPolicy
from convlab.util.multiwoz_slot_trans import REF_USR_DA, REF_SYS_DA
from convlab.modules.usr.multiwoz.uber_usr.data_loader import DataLoader
import numpy as np
import os
import copy

from convlab.modules.usr.multiwoz.uber_usr.model import E2EUser

DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'don\'t care'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK]

# import reflect table
REF_USR_DA_M = copy.deepcopy(REF_USR_DA)
REF_SYS_DA_M = {}
for dom, ref_slots in REF_SYS_DA.items():
    dom = dom.lower()
    REF_SYS_DA_M[dom] = {}
    for slot_a, slot_b in ref_slots.items():
        REF_SYS_DA_M[dom][slot_a.lower()] = slot_b
    REF_SYS_DA_M[dom]['none'] = None

# def book slot
BOOK_SLOT = ['people', 'day', 'stay', 'time']

class Goal(object):
    def __init__(self, goal_generator: GoalGenerator, seed=None):
        """
        create new Goal by random
        Args:
            goal_generator (GoalGenerator): Goal Gernerator.
        """
        self.domain_goals = goal_generator.get_user_goal(seed)
        self.domain_goals_org = copy.deepcopy(self.domain_goals)

        self.domains = list(self.domain_goals['domain_ordering'])
        del self.domain_goals['domain_ordering']

        for domain in self.domains:
            if 'reqt' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['reqt'] = {slot: DEF_VAL_UNK for slot in self.domain_goals[domain]['reqt']}

            if 'book' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['booked'] = DEF_VAL_UNK

            if domain in ['attraction', 'restaurant', 'hotel']:
                if 'name' not in self.domain_goals[domain].get('info', {}).keys():
                    old_dict = self.domain_goals[domain].get('reqt', {})
                    old_dict['name'] = DEF_VAL_UNK
                    old_dict['name'] = DEF_VAL_UNK
                    self.domain_goals[domain]['reqt'] = old_dict

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        for domain in self.domains:
            if 'reqt' in self.domain_goals[domain]:
                reqt_vals = self.domain_goals[domain]['reqt'].values()
                for val in reqt_vals:
                    if val in NOT_SURE_VALS:
                        return False

            if 'booked' in self.domain_goals[domain]:
                if self.domain_goals[domain]['booked'] in NOT_SURE_VALS:
                    return False
        return True

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'

class UserPolicyHUS(UserPolicy):


    def __init__(self):

        self.max_turn = 40
        self.max_response_len = 15

        self.goal_generator = GoalGenerator(corpus_path='data/multiwoz/annotated_user_da_with_span_full.json')

        self.goal = None
        self.sys_da = None
        self.usr_da = None
        self.session_over = False
        self.cur_domain = None

        self.data = DataLoader()
        self.voc_goal, self.voc_usr, self.voc_sys = self.data.vocab_loader()
        self.goal_vocab_size, self.usr_vocab_size, self.sys_vocab_size = self.data.get_voc_size()
        self.user = E2EUser(self.goal_vocab_size, self.usr_vocab_size, self.sys_vocab_size)
        model_path = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'usr'), 'uber_usr'), 'save')
        self.user.load_checkpoint(os.path.join(model_path, '9-00003500'))


    def init_session(self):

        self.__turn = 0
        self.session_over = False

        self.goal = Goal(self.goal_generator)
        self.sys_da = []
        self.__user_action = []
        self.cur_domain = None

    def predict(self, state, sys_action):
        self.__turn += 2
        if(self.__turn) >= self.max_turn:
            self.close_session()
            return {}, True, self.reward

        self.reward = -1

        self.sys_da.append(sys_action)

        self.sys_da_seq = [[self.data.sysda2seq(da, self.goal.domain_goals_org) for da in self.sys_da]]
        self.sys_da_seq = [[[self.voc_sys.get(word, self.voc_sys[self.data.unk]) for word in sys_da] for sys_da in sys_das] for
                           sys_das in self.sys_da_seq]
        self.goal_seq = [[self.voc_goal.get(word, self.voc_goal[self.data.unk]) for word in goal] for goal in [self.data.usrgoal2seq(self.goal.domain_goals_org)]]

        batch_input = {}
        posts_length = []
        posts = []
        goals_length = []
        goals = []

        ''' start padding '''
        sentence_num = [len(sess) for sess in self.sys_da_seq]
        max_sentence_num = max(sentence_num)

        max_goal_length = max([len(sess_goal) for sess_goal in self.goal_seq])
        for i, l in enumerate(sentence_num):
            goals_length += [len(self.goal_seq[i])] * l
            goals_padded = self.goal_seq[i] + [0] * (max_goal_length - len(self.goal_seq[i]))
            goals += [goals_padded] * l

        for sess in self.sys_da_seq:

            sen_padded = padding(sess, 15)

            for j, sen in enumerate(sess):
                if j == 0:
                    post_single = np.zeros([max_sentence_num, 15], np.int)
                    post_length_single = np.zeros([max_sentence_num], np.int)
                else:
                    post_single = posts[-1]
                    post_length_single = posts_length[-1]
                post_length_single[j] = min(len(sen), 15)
                post_single[j, :] = sen_padded[j]

                posts_length.append(post_length_single)
                posts.append(post_single)
        ''' end padding '''

        batch_input['posts_length'] = posts_length
        batch_input['posts'] = posts
        batch_input['goals_length'] = goals_length
        batch_input['goals'] = goals
        response_idx = self.user.inference(batch_input)
        response_seq = self.data.id2sentence(response_idx[0][0])
        user_action = self.data.usrseq2da(response_seq, self.goal.domain_goals_org)
        self.__user_action.append(user_action)

        # print("User action: ", user_action)
        self.update(user_action, self.goal.domain_goals)

        return user_action, self.session_over, self.reward

    def update(self, user_action, domain_goals):
        # print("Org goal: ", domain_goals)
        for user_act in user_action:
            domain, intent = user_act.split('-')
            if intent == 'Request':
                for slot in user_action[user_act]:
                    try:
                        del domain_goals[domain.lower()]['reqt'][REF_SYS_DA[domain].get(slot[0], slot[0])]
                    except:
                        pass
            elif intent == 'Inform':
                for slot in user_action[user_act]:
                    try:
                        del domain_goals[domain.lower()]['info'][REF_SYS_DA[domain].get(slot[0], slot[0])]
                    except:
                        pass
        # print("Goal after: ", domain_goals)


    def close_session(self):
        self.session_over = True
        if self.goal.task_complete == True:
            self.reward = 2.0 * self.max_turn
        else:
            self.reward = -1.0 * self.max_turn
        # print(self.goal.domain_goals)
        # print(self.goal.domain_goals_org)
        total_slot_cnt = 0
        satisfied_slot_cnt = 0
        for domain in self.goal.domain_goals_org:
            for intent in ['info', 'reqt']:
                try:
                    for slot in self.goal.domain_goals_org[domain][intent]:
                        total_slot_cnt += 1
                        if slot not in self.goal.domain_goals[domain][intent]:
                            satisfied_slot_cnt += 1
                except:
                    pass
        print("Total slots: ", total_slot_cnt)
        print("Satisfied slots: ", satisfied_slot_cnt)

def padding(origin, l):
    """
    pad a list of different lens "origin" to the same len "l"
    """
    new = origin.copy()
    for i, j in enumerate(new):
        new[i] += [0] * (l - len(j))
        new[i] = j[:l]
    return new