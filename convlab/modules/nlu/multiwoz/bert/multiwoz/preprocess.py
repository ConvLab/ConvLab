"""
Preprocess multiwoz data for BertNLU.

Usage:
    python preprocess [mode=all|usr|sys]
    mode: which side data will be use

Require:
    - ``../../../../../../data/multiwoz/[train|val|test].json.zip`` data file

Output:
    - ``data/[mode]_data/``: processed data dir
        - ``data.pkl``: data[data_key=train|val|test] is a list of [tokens,tags,intents],
            tokens: list of words; tags: list of BIO tags(e.g. B-domain-intent+slot); intent: list of 'domain-intent+slot*value'.
        - ``intent_vocab.pkl``: list of all intents (format: 'domain-intent+slot*value')
        - ``tag_vocab.pkl``: list of all tags (format: 'O'|'B-domain-intent+slot'|'B-domain-intent+slot')
"""
import json
import os
import zipfile
import sys
import pickle
from collections import Counter


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def preprocess(mode):
    assert mode == 'all' or mode == 'usr' or mode == 'sys'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '../../../../../../data/multiwoz')
    processed_data_dir = os.path.join(cur_dir, 'data/{}_data'.format(mode))
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ['train', 'val', 'test']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}
    all_da = []
    all_intent = []
    all_tag = []
    for key in data_key:
        processed_data[key] = []
        for no, sess in data[key].items():
            for is_sys, turn in enumerate(sess['log']):
                if mode == 'usr' and is_sys % 2 == 1:
                    continue
                elif mode == 'sys' and is_sys % 2 == 0:
                    continue
                tokens = turn["text"].split()
                dialog_act = {}
                for dacts in turn["span_info"]:
                    if dacts[0] not in dialog_act:
                        dialog_act[dacts[0]] = []
                    dialog_act[dacts[0]].append([dacts[1], " ".join(tokens[dacts[3]: dacts[4] + 1])])

                spans = turn["span_info"]
                tags = []
                for i, _ in enumerate(tokens):
                    for span in spans:
                        if i == span[3]:
                            tag = "B-" + span[0] + "+" + span[1]
                            if key != 'train' and tag not in all_tag:
                                tags.append('O')
                            else:
                                tags.append(tag)
                            break
                        if span[3] < i <= span[4]:
                            tag = "I-" + span[0] + "+" + span[1]
                            if key != 'train' and tag not in all_tag:
                                tags.append('O')
                            else:
                                tags.append(tag)
                            break
                    else:
                        tags.append("O")

                intents = []
                for dacts in turn["dialog_act"]:
                    for dact in turn["dialog_act"][dacts]:
                        if dacts not in dialog_act or dact[0] not in [sv[0] for sv in dialog_act[dacts]]:
                            if dact[1] in ["none", "?", "yes", "no", "do nt care", "do n't care"]:
                                intents.append(dacts + "+" + dact[0] + "*" + dact[1])
                processed_data[key].append([tokens, tags, intents, turn["dialog_act"]])
                if key == 'train':
                    all_da += [da for da in turn['dialog_act']]
                    all_intent += intents
                    all_tag += tags
        if key == 'train':
            all_da = [x[0] for x in dict(Counter(all_da)).items() if x[1]]
            all_intent = [x[0] for x in dict(Counter(all_intent)).items() if x[1]]
            all_tag = [x[0] for x in dict(Counter(all_tag)).items() if x[1]]

    for key in data_key:
        print('loaded {}, size {}'.format(key, len(processed_data[key])))
    print('dialog act num:', len(all_da))
    print('sentence label num:', len(all_intent))
    print('tag num:', len(all_tag))
    pickle.dump(processed_data, open(os.path.join(processed_data_dir, 'data.pkl'), 'wb'))
    pickle.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.pkl'), 'wb'))
    pickle.dump(all_tag, open(os.path.join(processed_data_dir, 'tag_vocab.pkl'), 'wb'))


if __name__ == '__main__':
    mode = sys.argv[1]
    preprocess(mode)
