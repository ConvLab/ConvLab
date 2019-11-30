"""
Based on pre-trained bert, BERTNLU use a linear layer for slot tagging and another linear layer for intent classification.
For more information, please refer to ``README.md``

Trained models can be download on:

- https://convlab.blob.core.windows.net/models/bert_multiwoz_all_context.zip

References:

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019, June). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 4171-4186).
"""
import os
import zipfile
import json
import pickle
import torch
from transformers import BertConfig
from unidecode import unidecode

from convlab.lib.file_util import cached_path
from convlab.modules.nlu.nlu import NLU
from convlab.modules.nlu.multiwoz.bert.dataloader import Dataloader
from convlab.modules.nlu.multiwoz.bert.jointBERT import JointBERT
from convlab.modules.nlu.multiwoz.bert.multiwoz.postprocess import recover_intent
from convlab.modules.nlu.multiwoz.bert.multiwoz.preprocess import preprocess


class BERTNLU(NLU):
    def __init__(self, mode, config_file, model_file):
        """
        BERT NLU initialization.

        Args:
            mode (str):
                can be either `'usr'`, `'sys'` or `'all'`, representing which side of data the model was trained on.

            model_file (str):
                model path or url

        Example:
            nlu = BERTNLU(mode='all', model_file='https://convlab.blob.core.windows.net/models/bert_multiwoz_all_context.zip')
        """
        assert mode == 'usr' or mode == 'sys' or mode == 'all'
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/{}'.format(config_file))
        config = json.load(open(config_file))
        DEVICE = config['DEVICE']
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])

        if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
            preprocess(mode)

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
        dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        print('intent num:', len(intent_vocab))
        print('tag num:', len(tag_vocab))

        bert_config = BertConfig.from_pretrained(config['model']['pretrained_weights'])

        best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        if not os.path.exists(best_model_path):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_dir)
            archive.close()
        print('Load from', best_model_path)
        model = JointBERT(bert_config, config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
        model.to(DEVICE)
        model.eval()

        self.model = model
        self.dataloader = dataloader
        print("BERTNLU loaded")

    def parse(self, utterance, context=[]):
        """
        Predict the dialog act of a natural language utterance.

        Args:
            utterance (str):
                A natural language utterance.

        Returns:
            output (dict):
                The dialog act of utterance.
        """
        ori_word_seq = unidecode(utterance).split()
        ori_tag_seq = ['O'] * len(ori_word_seq)
        context_seq = self.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-3:]))
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = self.dataloader.bert_tokenize(ori_word_seq, ori_tag_seq)
        batch_data = [[ori_word_seq, ori_tag_seq, intents, da, context_seq,
                       new2ori, word_seq, self.dataloader.seq_tag2id(tag_seq), self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader._pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        slot_logits, intent_logits = self.model.forward(word_seq_tensor, word_mask_tensor,
                                                        context_seq_tensor=context_seq_tensor,
                                                        context_mask_tensor=context_mask_tensor)
        intent = recover_intent(self.dataloader, intent_logits[0], slot_logits[0], tag_mask_tensor[0],
                                batch_data[0][0], batch_data[0][-4])
        dialog_act = {}
        for act, slot, value in intent:
            dialog_act.setdefault(act, [])
            dialog_act[act].append([slot, value])
        return dialog_act


if __name__ == '__main__':
    nlu = BERTNLU(mode='all', config_file='multiwoz_all_context.json', model_file='https://convlab.blob.core.windows.net/models/bert_multiwoz_all_context.zip')
    test_utterances = [
        "What type of accommodations are they. No , i just need their address . Can you tell me if the hotel has internet available ?",
        "What type of accommodations are they.",
        "No , i just need their address .",
        "Can you tell me if the hotel has internet available ?"
        "you're welcome! enjoy your visit! goodbye.",
        "yes. it should be moderately priced.",
        "i want to book a table for 6 at 18:45 on thursday",
        "i will be departing out of stevenage.",
        "What is the Name of attraction ?",
        "Can I get the name of restaurant?",
        "Can I get the address and phone number of the restaurant?",
        "do you have a specific area you want to stay in?"
    ]
    for utt in test_utterances:
        print(utt)
        print(nlu.parse(utt))
