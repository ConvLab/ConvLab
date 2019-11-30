import copy
import json
import os
import torch
import pickle
import urllib
import tarfile
from pprint import pprint
import random
from itertools import chain
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from convlab.modules.dst.state_tracker import Tracker
from convlab.modules.dst.multiwoz.dst_util import init_state, normalize_value
from convlab.modules.word_dst.multiwoz.sumbt import BeliefTracker
from convlab.modules.word_dst.multiwoz.sumbt.sumbt_utils import Processor, normalize_text
from convlab.modules.word_dst.multiwoz.sumbt.sumbt_config import *
from convlab.modules.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
USE_CUDA = False


def get_label_embedding(labels, max_seq_length, tokenizer, device):
    features = []
    for label in labels:
        label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
        label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
        label_len = len(label_token_ids)

        label_padding = [0] * (max_seq_length - len(label_token_ids))
        label_token_ids += label_padding
        assert len(label_token_ids) == max_seq_length

        features.append((label_token_ids, label_len))

    all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
    all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)

    return all_label_token_ids, all_label_len


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class SUMBTTracker(Tracker):
    """
    Transferable multi-domain dialogue state tracker, adopted from https://github.com/SKTBrain/SUMBT
    """

    def __init__(self, data_dir='data/sumbt', model_file=''):
        Tracker.__init__(self)


        if not os.path.exists(data_dir):
            if model_file == '':
                raise Exception(
                    'Please provide remote model file path in config')
            resp = urllib.request.urlretrieve(model_file)[0]
            temp_file = tarfile.open(resp)
            temp_file.extractall('data')
            assert os.path.exists(data_dir)

        processor = Processor(args)
        label_list = processor.get_labels()
        num_labels = [len(labels) for labels in label_list]  # number of slot-values in each slot-type

        # tokenizer
        vocab_dir = os.path.join(data_dir, 'model','%s-vocab.txt' % args.bert_model)
        if not os.path.exists(vocab_dir):
            raise ValueError("Can't find %s " % vocab_dir)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case=args.do_lower_case)
        num_train_steps = None
        accumulation = False
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.sumbt_model = BeliefTracker(args, num_labels, self.device)
        if args.fp16:
            self.sumbt_model.half()
        self.sumbt_model.to(self.device)

        ## Get slot-value embeddings
        self.label_token_ids, self.label_len = [], []
        for labels in label_list:
            token_ids, lens = get_label_embedding(labels, args.max_label_length, self.tokenizer, self.device)
            self.label_token_ids.append(token_ids)
            self.label_len.append(lens)
        self.label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
        self.label_map_inv = [{i: label for i, label in enumerate(labels)} for labels in label_list]
        self.label_list = label_list
        self.target_slot = processor.target_slot
        ## Get domain-slot-type embeddings
        self.slot_token_ids, self.slot_len = \
            get_label_embedding(processor.target_slot, args.max_label_length, self.tokenizer, self.device)

        self.args = args
        self.state = {}
        self.param_restored = False
        self.sumbt_model.initialize_slot_value_lookup(self.label_token_ids, self.slot_token_ids)
        self.load_param()
        self.det_dic = {}
        for domain, dic in REF_USR_DA.items():
            for key, value in dic.items():
                assert '-' not in key
                self.det_dic[key.lower()] = key + '-' + domain
                self.det_dic[value.lower()] = key + '-' + domain

        self.cached_res = {}

    def init_session(self):
        self.state = init_state()
        if not self.param_restored:
            self.load_param()
            self.param_restored = True

    def load_param(self):
        if USE_CUDA:
            self.sumbt_model.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model.bin")))
        else:
            self.sumbt_model.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model.bin"), map_location=lambda storage, loc: storage))

    def update(self, user_act=None):
        """Update the dialogue state with the generated tokens from TRADE"""
        if not isinstance(user_act, str):
            raise Exception(
                f'Expected user_act is str but found {type(user_act)}')
        prev_state = self.state

        actual_history = copy.deepcopy(prev_state['history'])
        actual_history[-1].append(user_act)  # [sys, user], [sys, user]
        query = self.construct_query(actual_history)
        pred_states = self.predict(query)

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        for state in pred_states:
            domain, slot, value = state.split('-', 2)
            value = '' if value == 'none' else value
            if slot not in ['name', 'book']:
                if domain not in new_belief_state:
                    if domain == 'bus':
                        continue
                    else:
                        raise Exception(
                            'Error: domain <{}> not in belief state'.format(domain))
            slot = REF_SYS_DA[domain.capitalize()].get(slot, slot)
            assert 'semi' in new_belief_state[domain]
            assert 'book' in new_belief_state[domain]
            if 'book' in slot:
                assert slot.startswith('book ')
                slot = slot.strip().split()[1]
            if slot == 'arrive by':
                slot = 'arriveBy'
            elif slot == 'leave at':
                slot = 'leaveAt'
            elif slot =='price range':
                slot = 'pricerange'
            domain_dic = new_belief_state[domain]
            if slot in domain_dic['semi']:
                new_belief_state[domain]['semi'][slot] = value
                    #normalize_value(self.value_dict, domain, slot, value)
            elif slot in domain_dic['book']:
                new_belief_state[domain]['book'][slot] = value
            elif slot.lower() in domain_dic['book']:
                new_belief_state[domain]['book'][slot.lower()] = value
            else:
                with open('trade_tracker_unknown_slot.log', 'a+') as f:
                    f.write(
                        f'unknown slot name <{slot}> with value <{value}> of domain <{domain}>\nitem: {state}\n\n')

        new_request_state = copy.deepcopy(prev_state['request_state'])
        # update request_state
        user_request_slot = self.detect_requestable_slots(user_act)
        for domain in user_request_slot:
            for key in user_request_slot[domain]:
                if domain not in new_request_state:
                    new_request_state[domain] = {}
                if key not in new_request_state[domain]:
                    new_request_state[domain][key] = user_request_slot[domain][key]

        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        new_state['request_state'] = new_request_state
        self.state = new_state
        # print((pred_states, query))
        return self.state

    def predict(self, query):
        cache_query_key = ''.join(str(list(chain.from_iterable(query[0]))))
        if cache_query_key in self.cached_res.keys():
            return self.cached_res[cache_query_key]

        input_ids, input_len = query
        input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
        input_len = torch.tensor(input_len).to(self.device).unsqueeze(0)
        labels = None
        _, pred_slot = self.sumbt_model(input_ids, input_len, labels)
        pred_slot_t = pred_slot[0][-1].tolist()
        predict_belief = []
        for idx,i in enumerate(pred_slot_t):
            predict_belief.append(f'{self.target_slot[idx]}-{self.label_map_inv[idx][i]}')
        self.cached_res[cache_query_key] = predict_belief
        return predict_belief

    def train(self):
        '''Training funciton of TRADE (to be added)'''
        pass

    def test(self):
        '''Testing funciton of TRADE (to be added)'''
        pass

    def construct_query(self, context):
        '''Construct query from context'''
        ids = []
        lens = []
        for sys_ut, user_ut in context:
            # utt_user = ''
            # utt_sys = ''
            if sys_ut.lower() == 'null':
                utt_sys = ''
            else:
                utt_sys = sys_ut

            utt_user = user_ut

            tokens_user = [x if x != '#' else '[SEP]' for x in self.tokenizer.tokenize(utt_user)]
            tokens_sys = [x if x != '#' else '[SEP]' for x in self.tokenizer.tokenize(utt_sys)]

            _truncate_seq_pair(tokens_user, tokens_sys, self.args.max_seq_length - 3)
            tokens = ["[CLS]"] + tokens_user + ["[SEP]"] + tokens_sys + ["[SEP]"]
            input_len = [len(tokens_user) + 2, len(tokens_sys) + 1]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.args.max_seq_length - len(input_ids))
            input_ids += padding
            assert len(input_ids) == self.args.max_seq_length
            ids.append(input_ids)
            lens.append(input_len)

        return (ids, lens)

    def detect_requestable_slots(self, observation):
        result = {}
        observation = observation.lower()
        _observation = ' {} '.format(observation)
        for value in self.det_dic.keys():
            _value = ' {} '.format(value.strip())
            if _value in _observation:
                key, domain = self.det_dic[value].split('-')
                if domain not in result:
                    result[domain] = {}
                result[domain][key] = 0
        return result


def test_update():
    # lower case, tokenized.
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sumbt_tracker = SUMBTTracker()
    sumbt_tracker.init_session()

    sumbt_tracker.state['history'] = [
        ['null', 'Could you book a 4 stars hotel for one night, 1 person?'],
        ['If you\'d like something cheap, I recommend the Allenbell']
    ]
    from timeit import default_timer as timer
    start = timer()
    pprint(sumbt_tracker.update('Friday and Can you book it for me and get a reference number ?'))
    sumbt_tracker.state['history'][-1].append('Friday and Can you book it for me and get a reference number ?')
    end = timer()
    print(end - start)
    #
    start = timer()
    sumbt_tracker.state['history'].append(['what is the area'])
    pprint(sumbt_tracker.update('in the east area of cambridge'))
    end = timer()
    print(end - start)

    start = timer()
    # sumbt_tracker.state['history'].append(['what is the area'])
    pprint(sumbt_tracker.update('in the east area of cambridge'))
    end = timer()
    print(end - start)


# if __name__ == '__main__':
#     test_update()