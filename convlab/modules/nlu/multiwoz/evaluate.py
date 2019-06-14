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

seed = 2019
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)


def da2triples(dialog_act):
    triples = []
    for intent, svs in dialog_act.items():
        for slot, value in svs:
            triples.append((intent, slot, value))
    return triples


if __name__ == '__main__':
    if len(sys.argv) != 2 :
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
    else:
        raise Exception("Available model: OneNetLU, MILU, SVMNLU")

    archive = zipfile.ZipFile('../../../../data/multiwoz/test.json.zip', 'r')
    test_data = json.load(archive.open('test.json'))
    TP, FP, FN = 0, 0, 0
    sen_num = 0
    sess_num = 0
    for no, session in test_data.items():
        sen_num += len(session['log'])
        sess_num += 1
        if sess_num%10==0:
            print('Session [%d|%d]' % (sess_num, len(test_data)))
            precision = 1.0 * TP / (TP + FP)
            recall = 1.0 * TP / (TP + FN)
            F1 = 2.0 * precision * recall / (precision + recall)
            print('Model {} on {} session {} sentences:'.format(model_name, sess_num, sen_num))
            print('\t Precision: %.2f' % (100 * precision))
            print('\t Recall: %.2f' % (100 * recall))
            print('\t F1: %.2f' % (100 * F1))
        for i, turn in enumerate(session['log']):
            labels = da2triples(turn['dialog_act'])
            predicts = da2triples(model.parse(turn['text']))
            for triple in predicts:
                if triple in labels:
                    TP += 1
                else:
                    FP += 1
            for triple in labels:
                if triple not in predicts:
                    FN += 1
    print(TP,FP,FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall)
    print('Model {} on {} session {} sentences:'.format(model_name,len(test_data),sen_num))
    print('\t Precision: %.2f' % (100 * precision))
    print('\t Recall: %.2f' % (100 * recall))
    print('\t F1: %.2f' % (100 * F1))
