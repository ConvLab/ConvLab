"""
Evaluate NLU models on Multiwoz test dataset
Metric: dataset level Precision/Recall/F1
Usage: PYTHONPATH=../../../.. python evaluate.py [OneNetLU|MILU|SVMNLU]
"""
import json
import random
import sys
import zipfile

import numpy
import torch

from convlab.modules.nlu.multiwoz import MILU
from convlab.modules.nlu.multiwoz import OneNetLU
from convlab.modules.nlu.multiwoz import SVMNLU
from convlab.modules.nlu.multiwoz import BERTNLU

seed = 2019
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)


def da2triples(dialog_act):
    triples = []
    for intent, svs in dialog_act.items():
        for slot, value in svs:
            triples.append((intent, slot, value.lower()))
    return triples


def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for ele in predicts:
            if ele in labels:
                TP += 1
            else:
                FP += 1
        for ele in labels:
            if ele not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, F1


def is_slot_da(da):
    tag_da = {'Inform', 'Select', 'Recommend', 'NoOffer', 'NoBook', 'OfferBook', 'OfferBooked', 'Book'}
    not_tag_slot = {'Internet', 'Parking', 'none'}
    if da[0].split('-')[1] in tag_da and da[1] not in not_tag_slot:
        return True
    return False


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:")
        print("\t python evaluate.py model_name")
        print("\t model_name=OneNetLU, MILU, or SVMNLU")
        sys.exit()
    model_name = sys.argv[1]
    if model_name == 'OneNetLU':
        model = OneNetLU(model_file="https://convlab.blob.core.windows.net/models/onenet.tar.gz")
    elif model_name == 'MILU':
        model = MILU(model_file="https://convlab.blob.core.windows.net/models/milu.tar.gz")
    elif model_name == 'SVMNLU':
        model = SVMNLU(model_file="https://convlab.blob.core.windows.net/models/svm_multiwoz.zip")
    elif model_name == 'BERTNLU':
        model = BERTNLU(mode='all', config_file='multiwoz_all_context.json', model_file='https://convlab.blob.core.windows.net/models/bert_multiwoz_all_context.zip')
    else:
        raise Exception("Available model: OneNetLU, MILU, SVMNLU")

    archive = zipfile.ZipFile('../../../../data/multiwoz/test.json.zip', 'r')
    test_data = json.load(archive.open('test.json'))
    sen_num = 0
    sess_num = 0
    predict_golden_intents = []
    predict_golden_slots = []
    predict_golden_all = []
    for no, session in test_data.items():
        sess_num += 1
        context = []
        for i, turn in enumerate(session['log']):
            if i % 2 == 1:
                # system action
                context.append(turn['text'])
                continue
            sen_num += 1
            labels = da2triples(turn['dialog_act'])
            predicts = da2triples(model.parse(turn['text'],context=context[-3:]))
            predict_golden_all.append({
                'predict': predicts,
                'golden': labels
            })
            predict_golden_slots.append({
                'predict': [x for x in predicts if is_slot_da(x)],
                'golden': [x for x in labels if is_slot_da(x)]
            })
            predict_golden_intents.append({
                'predict': [x for x in predicts if not is_slot_da(x)],
                'golden': [x for x in labels if not is_slot_da(x)]
            })
            context.append(turn['text'])
        if sess_num%100==0:
            precision, recall, F1 = calculateF1(predict_golden_all)
            print('Model {} on [{}|{}] session {} sentences:'.format(model_name, sess_num, len(test_data), sen_num))
            print('\t Precision: %.2f' % (100 * precision))
            print('\t Recall: %.2f' % (100 * recall))
            print('\t F1: %.2f' % (100 * F1))
            precision, recall, F1 = calculateF1(predict_golden_intents)
            print('-' * 20 + 'intent' + '-' * 20)
            print('\t Precision: %.2f' % (100 * precision))
            print('\t Recall: %.2f' % (100 * recall))
            print('\t F1: %.2f' % (100 * F1))
            precision, recall, F1 = calculateF1(predict_golden_slots)
            print('-' * 20 + 'slot' + '-' * 20)
            print('\t Precision: %.2f' % (100 * precision))
            print('\t Recall: %.2f' % (100 * recall))
            print('\t F1: %.2f' % (100 * F1))

    precision, recall, F1 = calculateF1(predict_golden_all)
    print('Model {} on {} session {} sentences:'.format(model_name,sess_num,sen_num))
    print('\t Precision: %.2f' % (100 * precision))
    print('\t Recall: %.2f' % (100 * recall))
    print('\t F1: %.2f' % (100 * F1))
    precision, recall, F1 = calculateF1(predict_golden_intents)
    print('-'*20+'intent'+'-'*20)
    print('\t Precision: %.2f' % (100 * precision))
    print('\t Recall: %.2f' % (100 * recall))
    print('\t F1: %.2f' % (100 * F1))
    precision, recall, F1 = calculateF1(predict_golden_slots)
    print('-' * 20 + 'slot' + '-' * 20)
    print('\t Precision: %.2f' % (100 * precision))
    print('\t Recall: %.2f' % (100 * recall))
    print('\t F1: %.2f' % (100 * F1))

