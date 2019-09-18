# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:29:52 2019

@author: truthless
"""
import os
import json
from convlab.modules.nlg.multiwoz.transformer.transformer import Constants
import logging
import torch

# input_ids, rep_in, resp_out, act_vecs

logger = logging.getLogger(__name__)

def get_batch(part, tokenizer, max_seq_length=50):
    examples = []
    data_dir = os.path.dirname(os.path.abspath(__file__))

    if part == 'train':
        with open('{}/data/train.json'.format(data_dir)) as f:
            source = json.load(f)
    elif part == 'val':
        with open('{}/data/val.json'.format(data_dir)) as f:
            source = json.load(f)
    elif part == 'test':
        with open('{}/data/test.json'.format(data_dir)) as f:
            source = json.load(f)
    else:
        raise ValueError(f'Unknown option {part}')

    logger.info("Loading total {} dialogs".format(len(source)))
    for num_dial, dialog_info in enumerate(source):
        hist = []
        dialog_file = dialog_info['file']
        dialog = dialog_info['info']
        
        for turn_num, turn in enumerate(dialog):

            tokens = tokenizer.tokenize(turn['user'])
            if len(hist) == 0:
                if len(tokens) > max_seq_length - 2:
                    tokens = tokens[:max_seq_length - 2]
            else:
                tokens = hist + [Constants.SEP_WORD] + tokens
                if len(tokens) > max_seq_length - 2:
                    tokens = tokens[-(max_seq_length - 2):]

            tokens = [Constants.CLS_WORD] + tokens + [Constants.SEP_WORD]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            padding = [Constants.PAD] * (max_seq_length - len(input_ids))
            input_ids += padding

            resp = [Constants.SOS_WORD] + tokenizer.tokenize(turn['sys']) + [Constants.EOS_WORD]

            if len(resp) > Constants.RESP_MAX_LEN:
                resp = resp[:Constants.RESP_MAX_LEN-1] + [Constants.EOS_WORD]
            else:
                resp = resp + [Constants.PAD_WORD] * (Constants.RESP_MAX_LEN - len(resp))

            resp_inp_ids = tokenizer.convert_tokens_to_ids(resp[:-1])
            resp_out_ids = tokenizer.convert_tokens_to_ids(resp[1:])

            act_vecs = [0] * len(Constants.act_ontology)
            for intent in turn['act']:
                for values in turn['act'][intent]:
                    w = intent + '-' + values[0] + '-' + values[1]
                    act_vecs[Constants.act_ontology.index(w)] = 1

            examples.append([input_ids, resp_inp_ids, resp_out_ids, act_vecs, dialog_file])

            sys = tokenizer.tokenize(turn['sys'])
            if turn_num == 0:
                hist = tokens[1:-1] + [Constants.SEP_WORD] + sys
            else:
                hist = hist + [Constants.SEP_WORD] + tokens[1:-1] + [Constants.SEP_WORD] + sys

    all_input_ids = torch.tensor([f[0] for f in examples], dtype=torch.long)
    all_response_in = torch.tensor([f[1] for f in examples], dtype=torch.long)
    all_response_out = torch.tensor([f[2] for f in examples], dtype=torch.long)
    all_act_vecs = torch.tensor([f[3] for f in examples], dtype=torch.float32)
    all_files = [f[4] for f in examples]

    return all_input_ids, all_response_in, all_response_out, all_act_vecs, all_files

def get_info(source, part):
    result = {}
    
    for num_dial, dialog_info in enumerate(source):
        dialog_file = dialog_info['file']
        dialog = dialog_info['info']
        result[dialog_file] = []
        
        for turn_num, turn in enumerate(dialog):
            result[dialog_file].append(turn[part])
            
    return result
