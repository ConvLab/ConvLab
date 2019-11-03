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
from convlab.modules.nlu.multiwoz.bert.model import BertNLU
from convlab.modules.nlu.multiwoz.bert.intentBERT import IntentBERT
from convlab.modules.nlu.multiwoz.bert.slotBERT import SlotBERT
from convlab.modules.nlu.multiwoz.bert.jointBERT import JointBERT
from convlab.modules.nlu.multiwoz.bert.multiwoz.postprocess import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser(description="Train a model.")
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
    for data_key in ['train', 'val', 'test']:
        dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)))), data_key)
        print('{} set size: {}'.format(data_key, len(dataloader.data[data_key])))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    bert_config = BertConfig.from_pretrained(config['model']['pretrained_weights'])

    model = JointBERT(bert_config, DEVICE, dataloader.tag_dim, dataloader.intent_dim, dataloader.intent_weight)
    model.to(DEVICE)

    for name, param in model.named_parameters():
        print(name, param.shape, param.device, param.requires_grad)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config['model']['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['model']['learning_rate'], eps=config['model']['adam_epsilon'])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config['model']['warmup_steps'], t_total=config['model']['max_step'])

    max_step = config['model']['max_step']
    check_step = config['model']['check_step']
    batch_size = config['model']['batch_size']
    model.zero_grad()
    set_seed(config['seed'])
    train_slot_loss, train_intent_loss = 0, 0
    best_val_f1 = 0.

    writer.add_text('config', json.dumps(config))

    for step in range(1, max_step + 1):
        model.train()
        batched_data = dataloader.get_train_batch(batch_size)
        batched_data = tuple(t.to(DEVICE) for t in batched_data)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor = batched_data
        _, _, slot_loss, intent_loss = model.forward(word_seq_tensor, word_mask_tensor, tag_seq_tensor, tag_mask_tensor,
                                                     intent_tensor)
        train_slot_loss += slot_loss.item()
        train_intent_loss += intent_loss.item()
        loss = slot_loss + intent_loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        if step % check_step == 0:
            train_slot_loss = train_slot_loss / check_step
            train_intent_loss = train_intent_loss / check_step
            print('[%d|%d] step' % (step, max_step))
            print('\t slot loss:', train_slot_loss)
            print('\t intent loss:', train_intent_loss)

            predict_golden_intents = []
            predict_golden_slots = []
            predict_golden_all = []

            val_slot_loss, val_intent_loss = 0, 0
            model.eval()
            for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key='val'):
                pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
                word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor = pad_batch

                with torch.no_grad():
                    slot_logits, intent_logits, slot_loss, intent_loss = model.forward(word_seq_tensor,
                                                                                       word_mask_tensor,
                                                                                       tag_seq_tensor,
                                                                                       tag_mask_tensor,
                                                                                       intent_tensor)
                val_slot_loss += slot_loss.item() * real_batch_size
                val_intent_loss += intent_loss.item() * real_batch_size
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

            for j in range(10):
                writer.add_text('val_sample_{}'.format(j), json.dumps(predict_golden_all[j]), global_step=step)

            total = len(dataloader.data['val'])
            val_slot_loss /= total
            val_intent_loss /= total
            print('%d samples val' % total)
            print('\t slot loss:', val_slot_loss)
            print('\t intent loss:', val_intent_loss)

            writer.add_scalars('intent_loss', {
                'train': train_intent_loss,
                'val': val_intent_loss,
                # 'test': test_intent_loss
            }, global_step=step)

            writer.add_scalars('slot_loss', {
                'train': train_slot_loss,
                'val': val_slot_loss,
                # 'test': test_tag_loss
            }, global_step=step)

            precision, recall, F1 = calculateF1(predict_golden_intents)
            print('-' * 20 + 'intent' + '-' * 20)
            print('\t Precision: %.2f' % (100 * precision))
            print('\t Recall: %.2f' % (100 * recall))
            print('\t F1: %.2f' % (100 * F1))

            writer.add_scalars('val_intent', {
                'precision': precision,
                'recall': recall,
                'F1': F1
            }, global_step=step)

            precision, recall, F1 = calculateF1(predict_golden_slots)
            print('-' * 20 + 'slot' + '-' * 20)
            print('\t Precision: %.2f' % (100 * precision))
            print('\t Recall: %.2f' % (100 * recall))
            print('\t F1: %.2f' % (100 * F1))

            writer.add_scalars('val_slot', {
                'precision': precision,
                'recall': recall,
                'F1': F1
            }, global_step=step)

            precision, recall, F1 = calculateF1(predict_golden_all)
            print('-' * 20 + 'overall' + '-' * 20)
            print('\t Precision: %.2f' % (100 * precision))
            print('\t Recall: %.2f' % (100 * recall))
            print('\t F1: %.2f' % (100 * F1))

            writer.add_scalars('val_overall', {
                'precision': precision,
                'recall': recall,
                'F1': F1
            }, global_step=step)

            if F1 > best_val_f1:
                best_val_f1 = F1
                model.save_pretrained(output_dir)
                print('best val F1 %.4f' % best_val_f1)
                print('save on', output_dir)

            train_slot_loss, train_intent_loss = 0, 0

    writer.close()

    model_path = os.path.join(output_dir, 'pytorch_model.bin')
    zip_path = config['zipped_model_path']
    print('zip model to', zip_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path)
