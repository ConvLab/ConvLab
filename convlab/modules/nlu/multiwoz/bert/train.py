import argparse
import pickle
import os
import json
from tatk.nlu.bert.dataloader import Dataloader
from tatk.nlu.bert.model import BertNLU
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import zipfile
from copy import deepcopy

torch.manual_seed(9102)
random.seed(9102)
np.random.seed(9102)


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

    data = pickle.load(open(os.path.join(data_dir,'data.pkl'),'rb'))
    intent_vocab = pickle.load(open(os.path.join(data_dir,'intent_vocab.pkl'),'rb'))
    tag_vocab = pickle.load(open(os.path.join(data_dir,'tag_vocab.pkl'),'rb'))
    for key in data:
        print('{} set size: {}'.format(key,len(data[key])))
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    dataloader = Dataloader(data, intent_vocab, tag_vocab, config['model']["pre-trained"])

    model = BertNLU(config['model'], dataloader.intent_dim, dataloader.tag_dim,
                    DEVICE=DEVICE,
                    intent_weight=dataloader.intent_weight)
    model.to(DEVICE)
    intent_save_params = []
    tag_save_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.device)
            if 'intent' in name:
                intent_save_params.append(name)
            elif 'tag' in name:
                tag_save_params.append(name)
            else:
                # Not support finetune bert params yet
                assert 0
    print('intent params:',intent_save_params)
    print('tag params:',tag_save_params)

    max_step = config['max_step']
    check_step = config['check_step']
    batch_size = config['batch_size']
    train_loss = 0
    train_intent_loss = 0
    train_tag_loss = 0
    best_val_intent_loss = np.inf
    best_val_tag_loss = np.inf
    best_intent_step = 0
    best_tag_step = 0
    best_intent_params = None
    best_tag_params = None

    for step in range(1,max_step+1):
        # batched_data = word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, word_seq_len
        batched_data = dataloader.get_train_batch(batch_size)
        intent_loss, tag_loss, total_loss, intent_logits, tag_logits = model.train_batch(*batched_data)
        train_intent_loss += intent_loss
        train_tag_loss += tag_loss
        train_loss += total_loss

        if step % check_step == 0:
            train_loss = train_loss / check_step
            train_intent_loss = train_intent_loss / check_step
            train_tag_loss = train_tag_loss / check_step
            print('[%d|%d] step train loss: %f' % (step, max_step, train_loss))
            print('\t intent loss:',train_intent_loss)
            print('\t tag loss:', train_tag_loss)

            val_loss = 0
            val_intent_loss = 0
            val_tag_loss = 0
            for batched_data, real_batch_size in dataloader.yield_batches(batch_size, data_key='val'):
                intent_loss, tag_loss, total_loss, intent_logits, tag_logits = model.eval_batch(*batched_data)
                val_intent_loss += intent_loss * real_batch_size
                val_tag_loss += tag_loss * real_batch_size
                val_loss += total_loss * real_batch_size
            total = len(dataloader.data['val'])
            val_loss /= total
            val_intent_loss /= total
            val_tag_loss /= total
            print('%d samples val loss: %f' % (total, val_loss))
            print('\t intent loss:', val_intent_loss)
            print('\t tag loss:', val_tag_loss)

            test_loss = 0
            test_intent_loss = 0
            test_tag_loss = 0
            for batched_data, real_batch_size in dataloader.yield_batches(batch_size, data_key='test'):
                intent_loss, tag_loss, total_loss, intent_logits, tag_logits = model.eval_batch(*batched_data)
                test_intent_loss += intent_loss * real_batch_size
                test_tag_loss += tag_loss * real_batch_size
                test_loss += total_loss * real_batch_size
            total = len(dataloader.data['test'])
            test_loss /= total
            test_intent_loss /= total
            test_tag_loss /= total
            print('%d samples test loss: %f' % (total, test_loss))
            print('\t intent loss:', test_intent_loss)
            print('\t tag loss:', test_tag_loss)

            update_flag = False
            if val_intent_loss < best_val_intent_loss:
                best_val_intent_loss = val_intent_loss
                best_intent_params = deepcopy({k: v for k, v in model.state_dict().items() if k in intent_save_params})
                update_flag = True
                best_intent_step = step
                print('best_intent_step', best_intent_step)
                print('best_val_intent_loss', best_val_intent_loss)
            if val_tag_loss < best_val_tag_loss:
                best_val_tag_loss = val_tag_loss
                best_tag_params = deepcopy({k: v for k, v in model.state_dict().items() if k in tag_save_params})
                update_flag = True
                best_tag_step = step
                print('best_tag_step', best_tag_step)
                print('best_val_tag_loss', best_val_tag_loss)

            if update_flag:
                print("Update best checkpoint")
                model_state_dict = {}
                model_state_dict.update(best_intent_params)
                model_state_dict.update(best_tag_params)
                best_model_path = os.path.join(output_dir, 'bestcheckpoint_step-{}.tar'.format(step))
                torch.save({
                    'best_intent_step': best_intent_step,
                    'best_tag_step': best_tag_step,
                    'step': step,
                    'model_state_dict': model_state_dict,
                    # 'optimizer_state_dict': model.optim.state_dict(),
                }, best_model_path)
                best_model_path = os.path.join(output_dir, 'bestcheckpoint.tar')
                torch.save({
                    'best_intent_step': best_intent_step,
                    'best_tag_step': best_tag_step,
                    'step': step,
                    'model_state_dict': model_state_dict,
                    # 'optimizer_state_dict': model.optim.state_dict(),
                }, best_model_path)
                print('save on', best_model_path)

            writer.add_scalars('total loss', {
                'train': train_loss,
                'val': val_loss,
                'test': test_loss
            }, global_step=step)

            writer.add_scalars('intent loss', {
                'train': train_intent_loss,
                'val': val_intent_loss,
                'test': test_intent_loss
            }, global_step=step)

            writer.add_scalars('tag loss', {
                'train': train_tag_loss,
                'val': val_tag_loss,
                'test': test_tag_loss
            }, global_step=step)

            train_loss = 0.0
            train_intent_loss = 0.0
            train_tag_loss = 0.0

    writer.close()

    model_path = os.path.join(output_dir, 'bestcheckpoint.tar')
    zip_path = config['zipped_model_path']
    print('zip model to', zip_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path)
