import numpy as np
import torch
import random
from pytorch_pretrained_bert import BertTokenizer
import math


class Dataloader:
    def __init__(self, data, intent_vocab, tag_vocab, tokenizer):
        """
        tokenize data and convert to ids.
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data: {'train': train_data, 'val': val_data, 'test': test_data}
        :param intent_vocab: list of all intens
        :param tag_vocab: list of all tags
        :param tokenizer: bert_tokenizer, same with pre-trained model
        """
        self.data = data
        self.intent_vocab = intent_vocab
        self.tag_vocab = tag_vocab
        self.intent_dim = len(intent_vocab)
        self.tag_dim = len(tag_vocab)
        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(tag_vocab)])
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.intent_weight = [0]*len(self.intent2id)
        for key in self.data:
            for d in self.data[key]:
                word_seq, tag_seq, new2ori = self.bert_tokenize(d[0], d[1])
                d.append(new2ori)
                d.append(word_seq)
                d.append(self.seq_tag2id(tag_seq))
                d.append(self.seq_intent2id(d[2]))
                if key=='train':
                    for intent_id in d[-1]:
                        self.intent_weight[intent_id] += 1
        train_size = len(self.data['train'])
        for intent, intent_id in self.intent2id.items():
            neg_pos = (train_size - self.intent_weight[intent_id]) / self.intent_weight[intent_id]
            # pos_weight param for intent classification. bigger->higher recall. Tune this on your dataset
            # 1) pos_weight = 1. low recall
            # self.intent_weight[intent_id] = 1
            # 2) pos_weight = neg_samples/pos_samples. predict too much
            # self.intent_weight[intent_id] = neg_pos
            # 3) pos_weight = log(neg_samples/pos_samples)
            self.intent_weight[intent_id] = np.log(neg_pos)
            # 4) pos_weight = min(MAX_WEIGHT, neg_pos)
            # self.intent_weight[intent_id] = min(20, neg_pos)
            # print(intent, self.intent_weight[intent_id], neg_pos)
        self.intent_weight = torch.tensor(self.intent_weight)

    def bert_tokenize(self, word_seq, tag_seq):
        split_tokens = []
        new_tag_seq = []
        new2ori = {}
        basic_tokens = self.tokenizer.basic_tokenizer.tokenize(' '.join(word_seq))
        accum = ''
        i, j = 0, 0
        for i, token in enumerate(basic_tokens):
            flag = (accum=='')
            if (accum+token).lower()==word_seq[j].lower():
                accum=''
            else:
                accum+=token
            first = True
            for sub_token in self.tokenizer.wordpiece_tokenizer.tokenize(basic_tokens[i]):
                new2ori[len(new_tag_seq)] = j
                split_tokens.append(sub_token)
                if flag and first:
                    new_tag_seq.append(tag_seq[j])
                    first=False
                else:
                    new_tag_seq.append('O')
            if accum=='':
                j += 1
        return split_tokens, new_tag_seq, new2ori

    def seq_tag2id(self, tags):
        return [self.tag2id[x] for x in tags if x in self.tag2id]

    def seq_id2tag(self, ids):
        return [self.id2tag[x] for x in ids]

    def seq_intent2id(self, intents):
        return [self.intent2id[x] for x in intents if x in self.intent2id]

    def seq_id2intent(self, ids):
        return [self.id2intent[x] for x in ids]

    def _pad_batch(self, batch_data):
        batch_size = len(batch_data)
        max_seq_len = max([len(x[-3]) for x in batch_data]) + 2
        word_seq_len = torch.zeros((batch_size), dtype=torch.long)
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.long)
        for i in range(batch_size):
            words = batch_data[i][-3]
            tags = batch_data[i][-2]
            intents = batch_data[i][-1]
            words = ['[CLS]'] + words + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(words)
            sen_len = len(words)
            word_seq_len[i] = sen_len
            word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens])
            tag_seq_tensor[i, 1:sen_len-1] = torch.LongTensor(tags)
            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)
            tag_mask_tensor[i, 1:sen_len-1] = torch.LongTensor([1] * (sen_len-2))
            for j in intents:
                intent_tensor[i, j] = 1
        return word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor

    def get_train_batch(self, batch_size):
        batch_data = random.choices(self.data['train'], k=batch_size)
        return self._pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i * batch_size:(i + 1) * batch_size]
            yield self._pad_batch(batch_data), len(batch_data)
