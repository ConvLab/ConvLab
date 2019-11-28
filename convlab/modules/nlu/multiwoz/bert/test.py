import argparse
import pickle
import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import zipfile
from copy import deepcopy
from pprint import pprint
from transformers import BertConfig, AdamW, WarmupLinearSchedule
from convlab.modules.nlu.multiwoz.bert.dataloader import Dataloader
from convlab.modules.nlu.multiwoz.bert.jointBERT import JointBERT
from convlab.modules.nlu.multiwoz.bert.multiwoz.postprocess import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser(description="Test a model.")
parser.add_argument('--config_path',
                    help='path to config file')


if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path))
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    log_dir = config['log_dir']
    DEVICE = config['DEVICE']

    intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
    tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
    dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                            pretrained_weights=config['model']['pretrained_weights'])
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))
    for data_key in ['val', 'test']:
        dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)))), data_key)
        print('{} set size: {}'.format(data_key, len(dataloader.data[data_key])))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    bert_config = BertConfig.from_pretrained(config['model']['pretrained_weights'])

    model = JointBERT(bert_config, config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
    model.to(DEVICE)
    model.eval()

    batch_size = config['model']['batch_size']

    for data_key in ['test']:
        predict_golden_intents = []
        predict_golden_slots = []
        predict_golden_all = []
        slot_loss, intent_loss = 0, 0
        for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key=data_key):
            pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
            word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
            if not config['model']['context']:
                context_seq_tensor, context_mask_tensor = None, None

            with torch.no_grad():
                slot_logits, intent_logits, batch_slot_loss, batch_intent_loss = model.forward(word_seq_tensor,
                                                                                               word_mask_tensor,
                                                                                               tag_seq_tensor,
                                                                                               tag_mask_tensor,
                                                                                               intent_tensor,
                                                                                               context_seq_tensor,
                                                                                               context_mask_tensor)
            slot_loss += batch_slot_loss.item() * real_batch_size
            intent_loss += batch_intent_loss.item() * real_batch_size
            for j in range(real_batch_size):
                predicts = recover_intent(dataloader, intent_logits[j], slot_logits[j], tag_mask_tensor[j],
                                          ori_batch[j][0], ori_batch[j][-4])
                predicts = [[x[0], x[1], x[2].lower()] for x in predicts]
                labels = ori_batch[j][3]

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

        total = len(dataloader.data[data_key])
        slot_loss /= total
        intent_loss /= total
        print('%d samples %s' % (total, data_key))
        print('\t slot loss:', slot_loss)
        print('\t intent loss:', intent_loss)

        precision, recall, F1 = calculateF1(predict_golden_all)
        print('-' * 20 + 'overall' + '-' * 20)
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
