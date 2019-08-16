#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division, print_function, unicode_literals

import json
import os
import pickle
import re
import shutil
import tempfile
import time
import zipfile
from copy import deepcopy

import numpy as np
import torch

from convlab.lib.file_util import cached_path
from convlab.modules.dst.multiwoz.dst_util import init_state
from convlab.modules.policy.system.policy import SysPolicy
from convlab.modules.word_policy.multiwoz.mdrg.model import Model
from convlab.modules.word_policy.multiwoz.mdrg.utils import util, dbquery, delexicalize
from convlab.modules.word_policy.multiwoz.mdrg.utils.nlp import normalize

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), 'data/nrg/mdrg')

class Args(object):
    pass
args = Args()
args.no_cuda = True
args.seed = 1 
args.no_models = 20 
args.original = DATA_PATH #'/home/sule/projects/research/multiwoz/model/model/'
args.dropout = 0. 
args.use_emb = 'False' 
args.beam_width = 10 
args.write_n_best = False 
args.model_path = os.path.join(DATA_PATH, 'translate.ckpt')
args.model_dir = DATA_PATH + '/'  #'/home/sule/projects/research/multiwoz/model/model/'
args.model_name = 'translate.ckpt'
args.valid_output = 'val_dials' #'/home/sule/projects/research/multiwoz/model/data/val_dials/'
args.test_output = 'test_dials' #'/home/sule/projects/research/multiwoz/model/data/test_dials/'

args.batch_size=64
args.vocab_size=400

args.use_attn=False
args.attention_type='bahdanau'
args.use_emb=False

args.emb_size=50
args.hid_size_enc=150
args.hid_size_dec=150
args.hid_size_pol=150
args.db_size=30
args.bs_size=94

args.cell_type='lstm'
args.depth=1
args.max_len=50

args.optim='adam'
args.lr_rate=0.005
args.lr_decay=0.0
args.l2_norm=0.00001
args.clip=5.0

args.teacher_ratio=1.0
args.dropout=0.0

args.no_cuda=True

args.seed=0
args.train_output='train_dials' #'data/train_dials/'

args.max_epochs=15
args.early_stop_count=2

args.load_param=False
args.epoch_load=0

args.mode='test'

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


def load_config(args):
    config = util.unicode_to_utf8(
        json.load(open('%s.json' % args.model_path, 'rb')))
    for key, value in args.__args.items():
        try:
            config[key] = value.value
        except:
            config[key] = value

    return config


def addBookingPointer(state, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if "book" in state['restaurant']:
        if "booked" in state['restaurant']['book']:
            if state['restaurant']['book']["booked"]:
                if "reference" in state['restaurant']['book']["booked"][0]:
                    rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if "book" in state['hotel']:
        if "booked" in state['hotel']['book']:
            if state['hotel']['book']["booked"]:
                if "reference" in state['hotel']['book']["booked"][0]:
                    hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if "book" in state['train']:
        if "booked" in  state['train']['book']:
            if state['train']['book']["booked"]:
                if "reference" in state['train']['book']["booked"][0]:
                    train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    # pprint(pointer_vector)
    return pointer_vector


def addDBPointer(state):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    db_results = {}
    num_entities = {} 
    for domain in domains:
        # entities = dbPointer.queryResultVenues(domain, {'metadata': state})
        entities = dbquery.query(domain, state[domain]['semi'].items())
        num_entities[domain] = len(entities)
        if len(entities) > 0: 
            # fields = dbPointer.table_schema(domain)
            # db_results[domain] = dict(zip(fields, entities[0]))
            db_results[domain] = entities[0]
        # pointer_vector = dbPointer.oneHotVector(len(entities), domain, pointer_vector)
        pointer_vector = oneHotVector(len(entities), domain, pointer_vector)

    return pointer_vector, db_results, num_entities 

def oneHotVector(num, domain, vector):
    """Return number of available entities for particular domain."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
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


def delexicaliseReferenceNumber(sent, state):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    for domain in domains:
        if state[domain]['book']['booked']:
            for slot in state[domain]['book']['booked'][0]:
                if slot == 'reference':
                    val = '[' + domain + '_' + slot + ']'
                else:
                    val = '[' + domain + '_' + slot + ']'
                key = normalize(state[domain]['book']['booked'][0][slot])
                sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                # try reference with hashtag
                key = normalize("#" + state[domain]['book']['booked'][0][slot])
                sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                # try reference with ref#
                key = normalize("ref#" + state[domain]['book']['booked'][0][slot])
                sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent


def get_summary_bstate(bstate):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = ['taxi', 'restaurant', 'hospital', 'hotel', 'attraction', 'train', 'police']
    summary_bstate = []
    for domain in domains:
        domain_active = False

        booking = []
        #print(domain,len(bstate[domain]['book'].keys()))
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if bstate[domain]['book']['booked']:
                    booking.append(1)
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] == 'dont care' or bstate[domain]['semi'][slot] == 'dontcare' or bstate[domain]['semi'][slot] == "don't care":
                slot_enc[1] = 1
            elif bstate[domain]['semi'][slot]:
                slot_enc[2] = 1
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
        else:
            summary_bstate += [0]

    # pprint(summary_bstate)
    #print(len(summary_bstate))
    assert len(summary_bstate) == 94
    return summary_bstate


def populate_template(template, top_results, num_results, state):
    active_domain = None if len(top_results.keys()) == 0 else list(top_results.keys())[0]
    template = template.replace('book [value_count] of them', 'book one of them')
    tokens = template.split()
    response = []
    for token in tokens:
        if token.startswith('[') and token.endswith(']'):
            domain = token[1:-1].split('_')[0]
            slot = token[1:-1].split('_')[1]
            if domain == 'train' and slot == 'id':
                slot = 'trainID'
            if domain in top_results and len(top_results[domain]) > 0 and slot in top_results[domain]:
                # print('{} -> {}'.format(token, top_results[domain][slot]))
                response.append(top_results[domain][slot])
            elif domain == 'value':
                if slot == 'count':
                    response.append(str(num_results))
                elif slot == 'place':
                    if 'arrive' in response:
                        for d in state: 
                            if d == 'history':
                                continue
                            if 'destination' in state[d]['semi']:
                                response.append(state[d]['semi']['destination'])
                                break
                    elif 'leave' in response:
                        for d in state: 
                            if d == 'history':
                                continue
                            if 'departure' in state[d]['semi']:
                                response.append(state[d]['semi']['departure'])
                                break
                    else:
                        try:
                            for d in state: 
                                if d == 'history':
                                    continue
                                for s in ['destination', 'departure']:
                                    if s in state[d]['semi']:
                                        response.append(state[d]['semi'][s])
                                        raise
                        except:
                            pass
                        else:
                            response.append(token)
                elif slot == 'time':
                    if 'arrive' in ' '.join(response[-3:]):
                        if active_domain is not None and 'arriveBy' in top_results[active_domain]:
                            # print('{} -> {}'.format(token, top_results[active_domain]['arriveBy']))
                            response.append(top_results[active_domain]['arriveBy'])
                            continue 
                        for d in state: 
                            if d == 'history':
                                continue
                            if 'arriveBy' in state[d]['semi']:
                                response.append(state[d]['semi']['arriveBy'])
                                break
                    elif 'leave' in ' '.join(response[-3:]):
                        if active_domain is not None and 'leaveAt' in top_results[active_domain]:
                            # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                            response.append(top_results[active_domain]['leaveAt'])
                            continue 
                        for d in state: 
                            if d == 'history':
                                continue
                            if 'leaveAt' in state[d]['semi']:
                                response.append(state[d]['semi']['leaveAt'])
                                break
                    elif 'book' in response:
                        if state['restaurant']['book']['time'] != "":
                            response.append(state['restaurant']['book']['time'])
                    else:
                        try:
                            for d in state: 
                                if d == 'history':
                                    continue
                                for s in ['arriveBy', 'leaveAt']:
                                    if s in state[d]['semi']:
                                        response.append(state[d]['semi'][s])
                                        raise
                        except:
                            pass
                        else:
                            response.append(token)
                else:
                    # slot-filling based on query results 
                    for d in top_results:
                        if slot in top_results[d]:
                            response.append(top_results[d][slot])
                            break
                    else:
                        # slot-filling based on belief state
                        for d in state:
                            if d == 'history':
                                continue
                            if slot in state[d]['semi']:
                                response.append(state[d]['semi'][slot])
                                break
                        else:
                            response.append(token)
            else:
                if domain == 'hospital':
                    if slot == 'phone':
                        response.append('01223216297')
                    elif slot == 'department':
                        response.append('neurosciences critical care unit')
                elif domain == 'police':
                    if slot == 'phone':
                        response.append('01223358966')
                    elif slot == 'name':
                        response.append('Parkside Police Station')
                    elif slot == 'address':
                        response.append('Parkside, Cambridge')
                elif domain == 'taxi':
                    if slot == 'phone':
                        response.append('01223358966')
                    elif slot == 'color':
                        response.append('white')
                    elif slot == 'type':
                        response.append('toyota')
                else:
                    # print(token)
                    response.append(token)
        else:
            response.append(token)

    try:
        response = ' '.join(response)
    except Exception as e:
        # pprint(response)
        raise
    response = response.replace(' -s', 's')
    response = response.replace(' -ly', 'ly')
    response = response.replace(' .', '.')
    response = response.replace(' ?', '?')
    return response


def mark_not_mentioned(state):
    for domain in state:
        # if domain == 'history':
        if domain not in ['police', 'hospital', 'taxi', 'train', 'attraction', 'restaurant', 'hotel']:
            continue
        try:
            # if len([s for s in state[domain]['semi'] if s != 'book' and state[domain]['semi'][s] != '']) > 0:
                # for s in state[domain]['semi']:
                #     if s != 'book' and state[domain]['semi'][s] == '':
                #         state[domain]['semi'][s] = 'not mentioned'
            for s in state[domain]['semi']:
                if state[domain]['semi'][s] == '':
                    state[domain]['semi'][s] = 'not mentioned'
        except Exception as e:
            # print(str(e))
            # pprint(state[domain])
            pass


def predict(model, prev_state, prev_active_domain, state, dic):
    start_time = time.time()
    model.beam_search = False
    input_tensor = []; bs_tensor = []; db_tensor = []

    usr = state['history'][-1][-1]

    prev_state = deepcopy(prev_state['belief_state'])
    state = deepcopy(state['belief_state'])

    mark_not_mentioned(prev_state)
    mark_not_mentioned(state)

    words = usr.split()
    usr = delexicalize.delexicalise(' '.join(words), dic)

    # parsing reference number GIVEN belief state
    usr = delexicaliseReferenceNumber(usr, state)

    # changes to numbers only here
    digitpat = re.compile('\d+')
    usr = re.sub(digitpat, '[value_count]', usr)
    # dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)

    # add database pointer
    pointer_vector, top_results, num_results = addDBPointer(state)
    # add booking pointer
    pointer_vector = addBookingPointer(state, pointer_vector)
    belief_summary = get_summary_bstate(state)

    tensor = [model.input_word2index(word) for word in normalize(usr).strip(' ').split(' ')] + [util.EOS_token] 
    input_tensor.append(torch.LongTensor(tensor))  
    bs_tensor.append(belief_summary) #
    db_tensor.append(pointer_vector) # db results and booking
    # bs_tensor.append([0.] * 94) #
    # db_tensor.append([0.] * 30) # db results and booking
    # create an empty matrix with padding tokens
    input_tensor, input_lengths = util.padSequence(input_tensor)
    bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
    db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

    output_words, loss_sentence = model.predict(input_tensor, input_lengths, input_tensor, input_lengths,
                                                db_tensor, bs_tensor)
    active_domain = get_active_domain(prev_active_domain, prev_state, state)
    if active_domain is not None and active_domain in num_results:
        num_results = num_results[active_domain]
    else:
        num_results = 0
    if active_domain is not None and active_domain in top_results:
        top_results = {active_domain: top_results[active_domain]}
    else:
        top_results = {} 
    response = populate_template(output_words[0], top_results, num_results, state)
    return response, active_domain


def get_active_domain(prev_active_domain, prev_state, state):
    domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi', 'hospital', 'police']
    active_domain = None
    # print('get_active_domain')
    for domain in domains:
        if domain not in prev_state and domain not in state:
            continue
        if domain in prev_state and domain not in state:
            return domain
        elif domain not in prev_state and domain in state:
            return domain
        elif prev_state[domain] != state[domain]:
            active_domain = domain
    if active_domain is None:
        active_domain = prev_active_domain
    return active_domain 


def loadModel(num):
    # Load dictionaries
    with open(os.path.join(DATA_PATH, 'input_lang.index2word.json')) as f:
        input_lang_index2word = json.load(f)
    with open(os.path.join(DATA_PATH, 'input_lang.word2index.json')) as f:
        input_lang_word2index = json.load(f)
    with open(os.path.join(DATA_PATH, 'output_lang.index2word.json')) as f:
        output_lang_index2word = json.load(f)
    with open(os.path.join(DATA_PATH, 'output_lang.word2index.json')) as f:
        output_lang_word2index = json.load(f)

    # Reload existing checkpoint
    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
    model.loadModel(iter=num)

    return model

DEFAULT_CUDA_DEVICE = -1
DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "milu.tar.gz")

class MDRGWordPolicy(SysPolicy):
    def __init__(self,
                archive_file=DEFAULT_ARCHIVE_FILE,
                cuda_device=DEFAULT_CUDA_DEVICE,
                model_file=None):
        
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for MDRG is specified!")
            archive_file = cached_path(model_file)

        temp_path = tempfile.mkdtemp()
        zip_ref = zipfile.ZipFile(archive_file, 'r')
        zip_ref.extractall(temp_path)
        zip_ref.close()

        self.dic = pickle.load(open(os.path.join(temp_path, 'mdrg/svdic.pkl'), 'rb'))
        # Load dictionaries
        with open(os.path.join(temp_path, 'mdrg/input_lang.index2word.json')) as f:
            input_lang_index2word = json.load(f)
        with open(os.path.join(temp_path, 'mdrg/input_lang.word2index.json')) as f:
            input_lang_word2index = json.load(f)
        with open(os.path.join(temp_path, 'mdrg/output_lang.index2word.json')) as f:
            output_lang_index2word = json.load(f)
        with open(os.path.join(temp_path, 'mdrg/output_lang.word2index.json')) as f:
            output_lang_word2index = json.load(f)
        self.response_model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
        self.response_model.loadModel(os.path.join(temp_path, 'mdrg/mdrg'))

        shutil.rmtree(temp_path)

        self.prev_state = init_state()
        self.prev_active_domain = None 

    def predict(self, state):
        try:
            response, active_domain = predict(self.response_model, self.prev_state, self.prev_active_domain, state, self.dic)
        except Exception as e:
            print('Response generation error', e)
            response = 'What did you say?'
            active_domain = None
        self.prev_state = deepcopy(state)
        self.prev_active_domain = active_domain
        return response



if __name__ == '__main__':
    dic = pickle.load(open(os.path.join(DATA_PATH, 'svdic.pkl'), 'rb'))
    state = {
        "police": {
            "book": {
                "booked": []
            },
            "semi": {}
        },
        "hotel": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "stay": ""
            },
            "semi": {
                "name": "",
                "area": "",
                "parking": "",
                "pricerange": "",
                "stars": "",
                "internet": "",
                "type": ""
            }
        },
        "attraction": {
            "book": {
                "booked": []
            },
            "semi": {
                "type": "",
                "name": "",
                "area": ""
            }
        },
        "restaurant": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "time": ""
            },
            "semi": {
                "food": "",
                "price range": "",
                "name": "",
                "area": "",
            }
        },
        "hospital": {
            "book": {
                "booked": []
            },
            "semi": {
                "department": ""
            }
        },
        "taxi": {
            "book": {
                "booked": []
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "departure": "",
                "arriveBy": ""
            }
        },
        "train": {
            "book": {
                "booked": [],
                "people": ""
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "day": "",
                "arriveBy": "",
                "departure": ""
            }
        }
    }

    m = loadModel(15)

    # modify state
    s = deepcopy(state)
    s['history'] = [['null', 'I want a korean restaurant in the centre.']]
    s['attraction']['semi']['area'] = 'centre'
    s['restaurant']['semi']['area'] = 'centre'
    s['restaurant']['semi']['food'] = 'korean'
    # s['history'] = [['null', 'i need to book a hotel in the east that has 4 stars.']]
    # s['hotel']['semi']['area'] = 'east'
    # s['hotel']['semi']['stars'] = '4'
    predict(m, state, s, dic)

    # import requests
    # resp = requests.post('http://localhost:10001', json={'history': [['null', 'I want a korean restaurant in the centre.']]})
    # if resp.status_code != 200:
    # #    raise Exception('POST /tasks/ {}'.format(resp.status_code))
    #     response = "Sorry, there is some problem" 
    # else:
    #     response = resp.json()["response"]
    # print('Response: {}'.format(response))
