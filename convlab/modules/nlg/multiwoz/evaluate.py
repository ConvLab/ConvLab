"""
Evaluate NLG models on sys utterances of Multiwoz test dataset
Metric: dataset level BLEU-4, slot error rate
Usage: PYTHONPATH=../../../.. python evaluate.py [SCLSTM|MultiwozTemplateNLG]
"""
import json
import random
import sys
import zipfile

import numpy
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from convlab.modules.nlg.multiwoz.multiwoz_template_nlg import MultiwozTemplateNLG
from convlab.modules.nlg.multiwoz.sc_lstm.nlg_sc_lstm import SCLSTM

seed = 2019
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)


def get_bleu4(dialog_acts, golden_utts, gen_utts):
    das2utts = {}
    for das, utt, gen in zip(dialog_acts, golden_utts, gen_utts):
        utt = utt.lower()
        gen = gen.lower()
        for da, svs in das.items():
            domain, act = da.split('-')
            if act == 'Request' or domain == 'general':
                continue
            else:
                for s, v in sorted(svs, key=lambda x: x[0]):
                    if s == 'Internet' or s == 'Parking' or s == 'none' or v == 'none':
                        continue
                    else:
                        v = v.lower()
                        if (' ' + v in utt) or (v + ' ' in utt):
                            utt = utt.replace(v, '{}-{}'.format(da, s), 1)
                        if (' ' + v in gen) or (v + ' ' in gen):
                            gen = gen.replace(v, '{}-{}'.format(da, s), 1)
        hash_key = ''
        for da in sorted(das.keys()):
            for s, v in sorted(das[da], key=lambda x: x[0]):
                hash_key += da + '-' + s + ';'
        das2utts.setdefault(hash_key, {'refs': [], 'gens': []})
        das2utts[hash_key]['refs'].append(utt)
        das2utts[hash_key]['gens'].append(gen)
    # pprint(das2utts)
    refs, gens = [], []
    for das in das2utts.keys():
        for gen in das2utts[das]['gens']:
            refs.append([s.split() for s in das2utts[das]['refs']])
            gens.append(gen.split())
    bleu = corpus_bleu(refs, gens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    return bleu


def get_err_slot(dialog_acts, nlg_model):
    assert isinstance(nlg_model, SCLSTM)
    errs = []
    N_total, p_total, q_total = 0, 0, 0
    for i, das in enumerate(dialog_acts):
        print('[%d/%d]'% (i+1,len(dialog_acts)))
        gen = nlg_model.generate_slots(das)
        triples = []
        counter = {}
        for da in das:
            if 'Request' in da or 'general' in da:
                continue
            for s,v in das[da]:
                if s == 'Internet' or s == 'Parking' or s == 'none' or v == 'none':
                        continue
                slot = da.lower()+'-'+s.lower()
                counter.setdefault(slot,0)
                counter[slot] += 1
                triples.append(slot+'-'+str(counter[slot]))
        assert len(set(triples))==len(triples)
        assert len(set(gen))==len(gen)
        N = len(triples)
        p = len(set(triples)-set(gen))
        q = len(set(gen)-set(triples))
        # print(triples)
        # print(gen)
        N_total+=N
        p_total+=p
        q_total+=q
        if N>0:
            err = (p+q)*1.0/N
            print(err)
            errs.append(err)
        # else:
            # assert q==0
        print('mean(std): {}({})'.format(np.mean(errs),np.std(errs)))
        if N_total>0:
            print('divide after sum:', (p_total+q_total)/N_total)
    return np.mean(errs)


if __name__ == '__main__':
    if len(sys.argv) != 2 :
        print("usage:")
        print("\t python evaluate.py model_name")
        print("\t model_name=SCLSTM or MultiwozTemplateNLG")
        sys.exit()
    model_name = sys.argv[1]
    print("Loading", model_name)
    if model_name == 'SCLSTM':
        model_sys = SCLSTM(model_file="https://convlab.blob.core.windows.net/models/nlg-sclstm-multiwoz.zip")
    elif model_name == 'MultiwozTemplateNLG':
        model_sys = MultiwozTemplateNLG(is_user=False)
    else:
        raise Exception("Available model: SCLSTM, MultiwozTemplateNLG")

    archive = zipfile.ZipFile('../../../../data/multiwoz/test.json.zip', 'r')
    test_data = json.load(archive.open('test.json'))

    dialog_acts = []
    golden_utts = []
    gen_utts = []

    sess_num = 0
    for no, sess in list(test_data.items()):
        sess_num+=1
        print('[%d/%d]' % (sess_num, len(test_data)))
        for i, turn in enumerate(sess['log']):
            if i % 2 == 0:
                continue
            dialog_acts.append(turn['dialog_act'])
            golden_utts.append(turn['text'])
            if model_name == 'SCLSTM':
                gen_utts.append(model_sys.generate(turn['dialog_act']))
            elif model_name == 'MultiwozTemplateNLG':
                gen_utts.append(model_sys.generate(turn['dialog_act']))

    bleu4 = get_bleu4(dialog_acts, golden_utts, gen_utts)

    print("Calculate bleu-4")
    print("BLEU-4: %.4f" % bleu4)

    if model_name == 'SCLSTM':
        print("Calculate slot error rate:")
        err = get_err_slot(dialog_acts, model_sys)
        print('ERR:', err)

    print("BLEU-4: %.4f" % bleu4)
    print(model_name)
