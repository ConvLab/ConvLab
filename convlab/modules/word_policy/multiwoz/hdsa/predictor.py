# -*- coding: utf-8 -*-
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on Thu Sep  5 17:00:52 2019

@author: truthless
"""

"""BERT finetuning runner."""

import os
import zipfile
import torch

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from convlab.lib.file_util import cached_path

def examine(domain, slot):
    if slot == "addr":
        slot = 'address'
    elif slot == "post":
        slot = 'postcode'
    elif slot == "ref":
        slot = 'ref'
    elif slot == "car":
        slot = "type"
    elif slot == 'dest':
        slot = 'destination'
    elif domain == 'train' and slot == 'id':
        slot = 'trainid'
    elif slot == 'leave':
        slot = 'leaveat'
    elif slot == 'arrive':
        slot = 'arriveby'
    elif slot == 'price':
        slot = 'pricerange'
    elif slot == 'depart':
        slot = 'departure'
    elif slot == 'name':
        slot = 'name'
    elif slot == 'type':
        slot = 'type'
    elif slot == 'area':
        slot = 'area'
    elif slot == 'parking':
        slot = 'parking'
    elif slot == 'internet':
        slot = 'internet'
    elif slot == 'stars':
        slot = 'stars'
    elif slot == 'food':
        slot = 'food'
    elif slot == 'phone':
        slot = 'phone'
    elif slot == 'day':
        slot = 'day'
    else:
        slot = 'illegal'
    return slot

def truncate_seq_pair(tokens_a, tokens_b, max_length):
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

class HDSA_predictor():
    def __init__(self, archive_file, model_file=None, use_cuda=False):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for DA-predictor is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(model_dir, 'checkpoints')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)
        
        load_dir = os.path.join(model_dir, "checkpoints/predictor/save_step_15120")
        if not os.path.exists(load_dir):
            archive = zipfile.ZipFile(f'{load_dir}.zip', 'r')
            archive.extractall(os.path.dirname(load_dir))
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
        self.max_seq_length = 256
        self.model = BertForSequenceClassification.from_pretrained(load_dir, 
            cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)), num_labels=44)
        self.device = 'cuda' if use_cuda else 'cpu'
        self.model.to(self.device)
    
    def _db_to_sentence(self, db_result, domain):
        if len(db_result)==0:
            return "no information"
        else:
            src = []
            for k, v in db_result.items():
                k = examine(domain, k.lower())
                if k != 'illegal' and isinstance(v, str):
                    src.extend([k, 'is', v])
            src = " ".join(src)
            return src

    def gen_feature(self,  usr_orig, sys_orig, db_result, domain):
        tokens_a = self.tokenizer.tokenize(usr_orig)
        tokens_b = self.tokenizer.tokenize(sys_orig)
        tokens_m = self.tokenizer.tokenize(self._db_to_sentence(db_result, domain))
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a) + 2)

        assert len(tokens) == len(segment_ids)

        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

        if len(tokens) < self.max_seq_length:
            if len(tokens_m) > self.max_seq_length - len(tokens) - 1:
                tokens_m = tokens_m[:self.max_seq_length - len(tokens) - 1]

            tokens += tokens_m + ['[SEP]']
            segment_ids += [0] * (len(tokens_m) + 1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        return input_ids, input_mask, segment_ids

    def predict(self, usr_orig, sys_orig, db_result, domain):
        
        input_ids, input_mask, segment_ids = self.gen_feature(usr_orig, sys_orig, db_result, domain)
        
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_masks = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_masks, labels=None)
            logits = torch.sigmoid(logits)
        preds = (logits > 0.4).float()
#        preds_numpy = preds.cpu().nonzero().squeeze().numpy()
#        
#        for i in preds_numpy:
#            if i < 10:
#                print(Constants.domains[i], end=' ')
#            elif i < 17:
#                print(Constants.functions[i-10], end=' ')
#            else:
#                print(Constants.arguments[i-17], end=' ')
#        print()
        
        return preds
