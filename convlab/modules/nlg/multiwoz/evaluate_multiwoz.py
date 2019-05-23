"""
evaluate NLG performance on multiwoz system side data
"""
import os
import sys

proj_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
sys.path.insert(0, proj_path)
print(sys.path[0])
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from convlab.modules.nlg.multiwoz.multiwoz_template_nlg import MultiwozTemplateNLG
from convlab.modules.nlg.multiwoz.sc_lstm.nlg_sc_lstm import SCLSTM
from convlab.modules.nlg.multiwoz.nlg import NLG
import json
import zipfile
import numpy as np


def get_BLEU4(da2utt_list, nlg_model):
    assert isinstance(nlg_model, NLG)
    das2utts = {}
    for das, utt in da2utt_list:
        utt = utt.lower()
        gen = nlg_model.generate(das).lower()
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
            for s,v in sorted(das[da], key=lambda x: x[0]):
                hash_key += da+'-'+s+';'
        das2utts.setdefault(hash_key,{'refs': [], 'gens': []})
        das2utts[hash_key]['refs'].append(utt)
        das2utts[hash_key]['gens'].append(gen)
    refs, gens = [], []
    for das in das2utts.keys():
        for gen in das2utts[das]['gens']:
            refs.append([s.split() for s in das2utts[das]['refs']])
            gens.append(gen.split())
    bleu = corpus_bleu(refs, gens, weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=SmoothingFunction().method1)
    return bleu


def get_err_slot(da2utt_list, nlg_model):
    assert isinstance(nlg_model, SCLSTM)
    errs = []
    N_total, p_total, q_total = 0, 0, 0
    for i, (das, utt) in enumerate(da2utt_list):
        print('[%d/%d]'% (i+1,len(da2utt_list)))
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
    return sum(errs)/len(errs)


if __name__ == '__main__':
    data_path = os.path.join(proj_path, 'data/multiwoz/test.json.zip')
    print(data_path)
    archive = zipfile.ZipFile(data_path, 'r')
    dataset = json.load(archive.open('test.json'))
    print('test set:', len(dataset))
    da2utt_list = []
    for no, sess in dataset.items():
        for i, turn in enumerate(sess['log']):
            if i % 2 == 1:
                da2utt_list.append((turn['dialog_act'], turn['text']))
    # print(da2utt_list[0])
    models = [MultiwozTemplateNLG(is_user=False), SCLSTM()]
    for model in models[1:]:
        # bleu4 = get_BLEU4(da2utt_list, model)
        # print(model, bleu4)
        # 0.33670454158214 for templateNLG
        # 0.4978659870379056 for sc-lstm
        err = get_err_slot(da2utt_list, model)
        print('ERR:',err)
