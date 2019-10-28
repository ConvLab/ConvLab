"""
Evaluate BertNLU models on multiwoz test dataset

Metric:
    dataset level Precision/Recall/F1

Usage:
    python evaluate.py [usr|sys|all]
"""
import pickle
import os
import json
import zipfile
from convlab.modules.nlu.multiwoz.bert.dataloader import Dataloader
from convlab.modules.nlu.multiwoz.bert.model import BertNLU
from convlab.modules.nlu.multiwoz.bert.multiwoz.postprocess import recover_intent
from convlab.lib.file_util import cached_path
import torch
import random
import numpy as np
import sys
from convlab.modules.nlu.multiwoz.bert.multiwoz.preprocess import preprocess

torch.manual_seed(9102)
random.seed(9102)
np.random.seed(9102)


if __name__ == '__main__':
    mode = sys.argv[1]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, 'multiwoz/configs/multiwoz_{}.json'.format(mode))
    config = json.load(open(config_path))
    DEVICE = config['DEVICE']
    data_dir = os.path.join(root_dir, config['data_dir'])
    output_dir = os.path.join(root_dir, config['output_dir'])
    log_dir = os.path.join(root_dir, config['log_dir'])

    if not os.path.exists(os.path.join(data_dir, 'data.pkl')):
        preprocess(mode)

    data = pickle.load(open(os.path.join(data_dir,'data.pkl'),'rb'))
    intent_vocab = pickle.load(open(os.path.join(data_dir,'intent_vocab.pkl'),'rb'))
    tag_vocab = pickle.load(open(os.path.join(data_dir,'tag_vocab.pkl'),'rb'))
    for key in data:
        print('{} set size: {}'.format(key,len(data[key])))
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))

    dataloader = Dataloader(data, intent_vocab, tag_vocab, config['model']["pre-trained"])

    best_model_path = os.path.join(output_dir, 'bestcheckpoint.tar')
    if not os.path.exists(best_model_path):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print('Load from zipped_model_path param')
        archive_file = cached_path(os.path.join(root_dir, config['zipped_model_path']))
        archive = zipfile.ZipFile(archive_file, 'r')
        archive.extractall(root_dir)
        archive.close()
    print('Load from', best_model_path)
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    print('best_intent_step', checkpoint['best_intent_step'])
    print('best_tag_step', checkpoint['best_tag_step'])

    model = BertNLU(config['model'], dataloader.intent_dim, dataloader.tag_dim,
                    DEVICE=DEVICE,
                    intent_weight=dataloader.intent_weight)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.to(DEVICE)
    model.eval()

    batch_size = config['batch_size']
    batch_num = len(dataloader.data['test']) // batch_size + 1

    golden_da_triples = []
    output_da_triples = []
    for i in range(batch_num):
        print("batch [%d|%d]" % (i + 1, batch_num))
        batch_data = dataloader.data['test'][i * batch_size:(i + 1) * batch_size]
        real_batch_size = len(batch_data)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor = dataloader._pad_batch(batch_data)
        intent_logits, tag_logits = model.predict_batch(word_seq_tensor, word_mask_tensor)
        for j in range(real_batch_size):
            intent = recover_intent(dataloader, intent_logits[j], tag_logits[j], tag_mask_tensor[j],
                                    batch_data[j][0], batch_data[j][-4])
            intent = [(x[0], x[1], x[2].lower()) for x in intent]
            output_da_triples.append(intent)
            triples = []
            for act, svs in batch_data[j][3].items():
                for s, v in svs:
                    triples.append((act, s, v.lower()))
            golden_da_triples.append(triples)

    TP, FP, FN = 0, 0, 0
    for (predicts, labels) in zip(output_da_triples, golden_da_triples):
        for triple in predicts:
            if triple in labels:
                TP += 1
            else:
                FP += 1
        for triple in labels:
            if triple not in predicts:
                FN += 1
    print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall)
    print('Model on {} sentences data_key={}'.format(len(dataloader.data['test']),mode))
    print('\t Precision: %.2f' % (100 * precision))
    print('\t Recall: %.2f' % (100 * recall))
    print('\t F1: %.2f' % (100 * F1))
    print('Load from', best_model_path)
    print('best_intent_step', checkpoint['best_intent_step'])
    print('best_tag_step', checkpoint['best_tag_step'])
