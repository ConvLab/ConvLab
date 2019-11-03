import argparse
import pickle
import os
import json
import torch
import random
import numpy as np
import zipfile

from convlab.modules.nlu.multiwoz.bert.dataloader import Dataloader
from convlab.modules.nlu.multiwoz.bert.model import BertNLU
from convlab.lib.file_util import cached_path

torch.manual_seed(9102)
random.seed(9102)
np.random.seed(9102)


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

    data = pickle.load(open(os.path.join(data_dir,'data.pkl'),'rb'))
    intent_vocab = pickle.load(open(os.path.join(data_dir,'intent_vocab.pkl'),'rb'))
    tag_vocab = pickle.load(open(os.path.join(data_dir,'tag_vocab.pkl'),'rb'))
    for key in data:
        print('{} set size: {}'.format(key,len(data[key])))
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))

    dataloader = Dataloader(data, intent_vocab, tag_vocab, config['model']["pre-trained"])

    best_model_path = best_model_path = os.path.join(output_dir, 'bestcheckpoint.tar')
    if not os.path.exists(best_model_path):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print('Load from zipped_model_path param')
        archive_file = cached_path(config['zipped_model_path'])
        archive = zipfile.ZipFile(archive_file, 'r')
        archive.extractall()
        archive.close()
    print('Load from',best_model_path)
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

    batch_size = config['batch_size']

    test_loss = 0
    test_intent_loss = 0
    test_tag_loss = 0
    for batched_data, _, real_batch_size in dataloader.yield_batches(batch_size, data_key='test'):
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
    print('Load from', best_model_path)
    print('best_intent_step', checkpoint['best_intent_step'])
    print('best_tag_step', checkpoint['best_tag_step'])
