import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import re
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import json
# import pandas as pd

from convlab.modules.word_dst.multiwoz.trade.trade_config import *

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

USE_CUDA = True if torch.cuda.is_available() else False
MAX_LENGTH = 10

args = {}
args['load_embedding'] = True
args["fix_embedding"] = False
if args["load_embedding"]:
    args["hidden"] = 400
    print("[Warning] Using hidden size = 400 for pretrained word embedding (300 + 100)...")

replacements = []
class TRADE(nn.Module):
    def __init__(self, hidden_size, lang, path, task, lr, dropout, slots, gating_dict, nb_train_vocab=0):
        super(TRADE, self).__init__()
        self.name = "TRADE"
        self.task = task
        self.hidden_size = hidden_size
        self.lang = lang[0]
        self.mem_lang = lang[1]
        self.lr = lr
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_temp = slots[2]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()
        self.data_dir = path

        self.encoder = EncoderRNN(
            self.lang.n_words, hidden_size, self.dropout, data_dir=path)
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout,
                                 self.slots, self.nb_gate)

        # if path:
        #     if USE_CUDA:
        #         print("MODEL {} LOADED".format(str(path)))
        #         trained_encoder = torch.load(str(path) + '/enc.th')
        #         trained_decoder = torch.load(str(path) + '/dec.th')
        #     else:
        #         print("MODEL {} LOADED".format(str(path)))
        #         trained_encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
        #         trained_decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)

        #     self.encoder.load_state_dict(trained_encoder)
        #     self.decoder.load_state_dict(trained_decoder)

        # Initialize optimizers and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)

        self.reset()
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_class = self.loss_class / self.print_every
        # print_loss_domain = self.loss_domain / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg, print_loss_ptr, print_loss_gate)

    def save_model(self, dec_type):
        directory = 'save/TRADE-' + args["addName"] + args['dataset'] + str(self.task) + '/' + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def train_batch(self, data, clip, slot_temp, reset=0):
        if reset:
            self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]
        all_point_outputs, gates, words_point_out, words_class_out = self.encode_and_decode(data, use_teacher_forcing,
                                                                                            slot_temp)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            # [:,:len(self.point_slots)].contiguous(),
            data["generate_y"].contiguous(),
            data["y_lengths"])  # [:,:len(self.point_slots)])
        loss_gate = self.cross_entorpy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)),
                                       data["gating_label"].contiguous().view(-1))

        if args["use_gate"]:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr

        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

    def optimize(self, clip):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def optimize_GEM(self, clip):
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp):
        # Build unknown mask for memory to encourage generalization
        if args['unk_mask'] and self.decoder.training:
            story_size = data['context'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial(
                [np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.Tensor(rand_mask)
            if USE_CUDA:
                rand_mask = rand_mask.cuda()
            story = data['context'] * rand_mask.long()
        else:
            story = data['context']

        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(
            story.transpose(0, 1), data['context_len'])

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(
            2) if self.encoder.training else 10
        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size,
                                                                                                     encoded_hidden,
                                                                                                     encoded_outputs,
                                                                                                     data[
                                                                                                         'context_len'],
                                                                                                     story, max_res_len,
                                                                                                     data['generate_y'],
                                                                                                     use_teacher_forcing,
                                                                                                     slot_temp)
        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def evaluate(self, dev, matric_best, slot_temp, early_stop=None):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k)
                                     for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            _, gates, words, class_words = self.encode_and_decode(
                data_dev, False, slot_temp)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {
                    "turn_belief": data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                # pointer-generator results
                if args["use_gate"]:
                    for si, sg in enumerate(gate):
                        if sg == self.gating_dict["none"]:
                            continue
                        elif sg == self.gating_dict["ptr"]:
                            pred = np.transpose(words[si])[bi]
                            st = []
                            for e in pred:
                                if e == 'EOS':
                                    break
                                else:
                                    st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(
                                    slot_temp[si] + "-" + str(st))
                        else:
                            predict_belief_bsz_ptr.append(
                                slot_temp[si] + "-" + inverse_unpoint_slot[sg.item()])
                else:
                    for si, _ in enumerate(gate):
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS':
                                break
                            else:
                                st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(
                                slot_temp[si] + "-" + str(st))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"]
                                                   [bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
                    print("True", set(data_dev["turn_belief"][bi]))
                    print("Pred", set(predict_belief_bsz_ptr), "\n")

        if args["genSample"]:
            json.dump(all_prediction, open(
                "all_prediction_{}.json".format(self.name), 'w'), indent=4)

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr",
                                                                                      slot_temp)

        evaluation_metrics = {"Joint Acc": joint_acc_score_ptr, "Turn Acc": turn_acc_score_ptr,
                              "Joint F1": F1_score_ptr}
        print(evaluation_metrics)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        # (joint_acc_score_ptr + joint_acc_score_class)/2
        joint_acc_score = joint_acc_score_ptr
        F1_score = F1_score_ptr

        if (early_stop == 'F1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")
            return F1_score
        else:
            if (joint_acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(joint_acc_score))
                print("MODEL SAVED")
            return joint_acc_score

    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(
                    set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(
                    set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total != 0 else 0
        turn_acc_score = turn_acc / float(total) if total != 0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / \
                float(precision + recall) if (precision + recall) != 0 else 0
        else:
            if len(pred) == 0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count
    
    def read_replacements(self):
        mapping_pair_url = os.path.join(self.data_dir, 'mapping.pair')
        fin_mapping_pair = open(mapping_pair_url, 'r')
        global replacements
        for line in fin_mapping_pair.readlines():
            tok_from, tok_to = line.replace('\n', '').split('\t')
            replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1, data_dir=''):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        if args["load_embedding"]:
            with open(os.path.join(data_dir, 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad",
                  self.embedding.weight.requires_grad)

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(2, bsz, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=False)
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3 * hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches,
                use_teacher_forcing, slot_temp):
        all_point_outputs = torch.zeros(
            len(slot_temp), batch_size, max_res_len, self.vocab_size)
        all_gate_outputs = torch.zeros(
            len(slot_temp), batch_size, self.nb_gate)
        if USE_CUDA:
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()

        # Get the slot embedding
        slot_emb_dict = {}
        for slot in slot_temp:
            # Domain embbeding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                if USE_CUDA:
                    domain_w2idx = domain_w2idx.cuda()
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                if USE_CUDA:
                    slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            slot_emb_dict[slot] = domain_emb + slot_emb

        words_class_out = []
        # Compute pointer-generator output
        words_point_out = []
        counter = 0
        for slot in slot_temp:  # TODO: Parallel this part to make it train faster
            hidden = encoded_hidden
            words = []
            slot_emb = slot_emb_dict[slot]
            decoder_input = self.dropout_layer(
                slot_emb).expand(batch_size, self.hidden_size)
            for wi in range(max_res_len):
                dec_state, hidden = self.gru(
                    decoder_input.expand_as(hidden), hidden)
                context_vec, logits, prob = self.attend(
                    encoded_outputs, hidden.squeeze(0), encoded_lens)
                if wi == 0:
                    all_gate_outputs[counter] = self.W_gate(context_vec)
                p_vocab = self.attend_vocab(
                    self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat(
                    [dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                p_context_ptr = torch.zeros(p_vocab.size())
                if USE_CUDA:
                    p_context_ptr = p_context_ptr.cuda()
                p_context_ptr.scatter_add_(1, story, prob)
                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                words.append([self.lang.index2word[w_idx.item()]
                              for w_idx in pred_word])
                all_point_outputs[counter, :, wi, :] = final_p_vocab
                if use_teacher_forcing:
                    decoder_input = self.embedding(
                        target_batches[:, counter, wi])  # Chosen word is next input
                else:
                    decoder_input = self.embedding(pred_word)
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()
            counter += 1
            words_point_out.append(words)
        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = F.softmax(scores_, dim=1)
        return scores


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def cross_entropy(logits, target):
    batch_size = logits.size(0)
    log_probs_flat = functional.log_softmax(logits)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target)
    loss = losses_flat.sum() / batch_size
    return loss


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))

    # logits_flat: (batch * max_len, num_classes)
    # -1 means infered from other dimentions
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def masked_binary_cross_entropy(logits, target, length):
    '''
    logits: (batch, max_len, num_class)
    target: (batch, max_len, num_class)
    '''
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))
    bce_criterion = nn.BCEWithLogitsLoss()
    loss = 0
    for bi in range(logits.size(0)):
        for i in range(logits.size(1)):
            if i < length[bi]:
                loss += bce_criterion(logits[bi][i], target[bi][i])
    loss = loss / length.float().sum()
    return loss


def masked_cross_entropy_(logits, target, length, take_log=False):
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))

    # logits_flat: (batch * max_len, num_classes)
    # -1 means infered from other dimentions
    logits_flat = logits.view(-1, logits.size(-1))
    if take_log:
        logits_flat = torch.log(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(logits_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def masked_coverage_loss(coverage, attention, length):
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))
    mask = sequence_mask(sequence_length=length)
    min_ = torch.min(coverage, attention)
    mask = mask.unsqueeze(2).expand_as(min_)
    min_ = min_ * mask.float()
    loss = min_.sum() / (len(length)*1.0)
    return loss


def masked_cross_entropy_for_slot(logits, target, mask, use_softmax=True):
    # print("logits", logits)
    # print("target", target)
    # -1 means infered from other dimentions
    logits_flat = logits.view(-1, logits.size(-1))
    # print(logits_flat.size())
    if use_softmax:
        log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    else:
        log_probs_flat = logits_flat  # torch.log(logits_flat)
    # print("log_probs_flat", log_probs_flat)
    target_flat = target.view(-1, 1)
    # print("target_flat", target_flat)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())  # b * |s|
    losses = losses * mask.float()
    loss = losses.sum() / (losses.size(0)*losses.size(1))
    # print("loss inside", loss)
    return loss


def masked_cross_entropy_for_value(logits, target, mask):
    # logits: b * |s| * m * |v|
    # target: b * |s| * m
    # mask:   b * |s|
    # -1 means infered from other dimentions
    logits_flat = logits.view(-1, logits.size(-1))
    # print(logits_flat.size())
    log_probs_flat = torch.log(logits_flat)
    # print("log_probs_flat", log_probs_flat)
    target_flat = target.view(-1, 1)
    # print("target_flat", target_flat)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())  # b * |s| * m
    loss = masking(losses, mask)
    return loss


def masking(losses, mask):
    mask_ = []
    batch_size = mask.size(0)
    max_len = losses.size(2)
    for si in range(mask.size(1)):
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if mask[:, si].is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = mask[:, si].unsqueeze(
            1).expand_as(seq_range_expand)
        mask_.append((seq_range_expand < seq_length_expand))
    mask_ = torch.stack(mask_)
    mask_ = mask_.transpose(0, 1)
    if losses.is_cuda:
        mask_ = mask_.cuda()
    losses = losses * mask_.float()
    loss = losses.sum() / (mask_.sum().float())
    return loss


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS",
                           EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split():
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
                for v in value.split():
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def normalize_text(text):
    global replacements
    # lower case every word
    text = text.lower()
    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text
