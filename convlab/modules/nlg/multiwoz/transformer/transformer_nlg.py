# -*- coding: utf-8 -*-

import re
import os
import zipfile
import json
import torch
import pickle
from copy import deepcopy
from convlab.lib.file_util import cached_path
from convlab.modules.nlg.nlg import NLG
from convlab.modules.word_policy.multiwoz.hdsa.tools import Tokenizer
from convlab.modules.word_policy.multiwoz.hdsa.transformer import Constants
from convlab.modules.word_policy.multiwoz.hdsa.transformer.Transformer import TransformerDecoder

timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")

DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "transformer.zip")

def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text

def normalize(text, sub=True):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    if sub:
        text = re.sub(timepat, ' [value_time] ', text)
        text = re.sub(pricepat, ' [train_price] ', text)
        #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?

    return utt

def delexicaliseReferenceNumber(sent, turn):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    for domain in turn:
        if turn[domain]['book']['booked']:
            for slot in turn[domain]['book']['booked'][0]:
                if slot == 'reference':
                    val = '[' + domain + '_' + slot + ']'
                else:
                    val = '[' + domain + '_' + slot + ']'
                key = normalize(turn[domain]['book']['booked'][0][slot])
                sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                # try reference with hashtag
                key = normalize("#" + turn[domain]['book']['booked'][0][slot])
                sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                # try reference with ref#
                key = normalize("ref#" + turn[domain]['book']['booked'][0][slot])
                sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent

class Transformer(NLG):
    
    def __init__(self, 
                 archive_file=DEFAULT_ARCHIVE_FILE, 
                 use_cuda=False,
                 model_file=None):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for Transformer is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(model_dir, 'checkpoints')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)
            
        with open(os.path.join(model_dir, "data/vocab.json"), 'r') as f:
            vocabulary = json.load(f)
        
        vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
        self.tokenizer = Tokenizer(vocab, ivocab)
        self.max_seq_length = 50
            
        self.decoder = TransformerDecoder(vocab_size=self.tokenizer.vocab_len, d_word_vec=128, act_dim=len(Constants.act_ontology), 
                                          n_layers=3, d_model=128, n_head=4, dropout=0.2)
        self.device = 'cuda' if use_cuda else 'cpu'
        self.decoder.to(self.device)
        checkpoint_file = os.path.join(model_dir, "checkpoints/transformer")
        self.decoder.load_state_dict(torch.load(checkpoint_file))
        
        with open(os.path.join(model_dir, 'data/svdic.pkl'), 'rb') as f:
            self.dic = pickle.load(f)

    def generate(self, meta, state):
        """
        meta = {"Attraction-Inform": [["Choice","many"],["Area","centre of town"]],
                "Attraction-Select": [["Type","church"],["Type"," swimming"],["Type"," park"]]}
        """
        usr_post = state['history'][-1][-1]
        usr = delexicalise(' '.join(usr_post.split()), self.dic)
    
        # parsing reference number GIVEN belief state
        usr = delexicaliseReferenceNumber(usr, state['belief_state'])
    
        # changes to numbers only here
        digitpat = re.compile('\d+')
        usr = re.sub(digitpat, '[value_count]', usr)
        
        tokens = self.tokenizer.tokenize(usr)
        if self.history:
            tokens = self.history + [Constants.SEP_WORD] + tokens
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[-(self.max_seq_length - 2):]
        tokens = [Constants.CLS_WORD] + tokens + [Constants.SEP_WORD]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # add placeholder value
        meta = deepcopy(meta)
        for k, v in meta.items():
            domain, intent = k.split('-')
            if intent == "Request":
                for pair in v:
                    if not isinstance(pair[1], str):
                        pair[1] = str(pair[1])
                    pair.insert(1, '?')
            else:
                counter = {}
                for pair in v:
                    if not isinstance(pair[1], str):
                        pair[1] = str(pair[1])
                    if pair[0] == 'Internet' or pair[0] == 'Parking':
                        pair.insert(1, 'yes')
                    elif pair[0] == 'none':
                        pair.insert(1, 'none')
                    else:
                        if pair[0] in counter:
                            counter[pair[0]] += 1
                        else:
                            counter[pair[0]] = 1
                        pair.insert(1, str(counter[pair[0]]))
        
        act_vecs = [0] * len(Constants.act_ontology)
        for intent in meta:
            for values in meta[intent]:
                w = intent + '-' + values[0] + '-' + values[1]
                if w in Constants.act_ontology:
                    act_vecs[Constants.act_ontology.index(w)] = 1
                    
        act_vecs = torch.tensor([act_vecs], dtype=torch.long).to(self.device)
        
        hyps = self.decoder.translate_batch(act_vecs=act_vecs, src_seq=input_ids, 
                                       n_bm=2, max_token_seq_len=40)
        pred = self.tokenizer.convert_id_to_tokens(hyps[0])
        
        if not self.history:
            self.history = tokens[1:-1] + [Constants.SEP_WORD] + self.tokenizer.tokenize(pred)
        else:
            self.history = self.history + [Constants.SEP_WORD] + tokens[1:-1] + [Constants.SEP_WORD] + self.tokenizer.tokenize(pred)
        
        # replace the placeholder with entities
        words = pred.split(' ')
        counter = {}
        for i in range(len(words)):
            if "[" in words[i] and "]" in words[i]:
                domain, slot = words[i].split('_')
                domain = domain[1:].capitalize()
                slot = slot[:-1].capitalize()
                key = '-'.join((domain, slot))
                flag = False
                for intent in meta:
                    _domain, _intent = intent.split('-')
                    if domain == _domain and _intent in ['Inform', 'Recommend', 'Offerbook']:
                        for values in meta[intent]:
                            if (slot == values[0]) and ('none' != values[-1]) and ((key not in counter) or (counter[key] == int(values[1])-1)):
                                words[i] = values[-1]
                                counter[key] = int(values[1])
                                flag = True
                                break
                        if flag:
                            break
        return " ".join(words)
