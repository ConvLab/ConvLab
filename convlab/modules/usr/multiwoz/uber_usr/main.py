import sys
from data_loader import DataLoader, batch_iter
from model import E2EUser
import argparse
import tensorflow as tf
import numpy as np
import datetime
import os
import random

BATCH_SIZE = 64

'''
input_graph = tf.Graph()
with input_graph.as_default():
    data = DataLoader()
    proto = tf.ConfigProto()
    input_sess = tf.Session(config=proto)
'''

data = DataLoader()
seq_goals, seq_usr_dass, seq_sys_dass = data.data_loader_seg()
train_goals, train_usrdas, train_sysdas, test_goals, test_usrdas, test_sysdas, val_goals, val_usrdas, val_sysdas = DataLoader.train_test_val_split_seg(
        seq_goals, seq_usr_dass, seq_sys_dass)
generator = batch_iter(train_goals, train_usrdas, train_sysdas, BATCH_SIZE)
generator_valid = batch_iter(val_goals, val_usrdas, val_sysdas, BATCH_SIZE)
generator_test = batch_iter(test_goals, test_usrdas, test_sysdas, BATCH_SIZE)

voc_goal_size, voc_usr_size, voc_sys_size = data.get_voc_size()

model = E2EUser(voc_goal_size, voc_usr_size, voc_sys_size)
model.load_checkpoint('save/0-00000060')


def padding(origin, l):
    """
    pad a list of different lens "origin" to the same len "l"
    """
    new = origin.copy()
    for i, j in enumerate(new):
        new[i] += [0] * (l - len(j))
        new[i] = j[:l]
    return new

def train(batch_goals, batch_usrdas, batch_sysdas):
    batch_input = {}
    posts_length = []
    posts = []
    origin_responses = []
    origin_responses_length = []
    goals_length = []
    goals = []

    ''' start padding '''
    sentence_num = [len(sess) for sess in batch_usrdas]
    max_sentence_num = max(sentence_num)
    
    max_goal_length = max([len(sess_goal) for sess_goal in batch_goals])
    for i, l in enumerate(sentence_num):
        goals_length += [len(batch_goals[i])] * l
        goals_padded = batch_goals[i] + [0] * (max_goal_length - len(batch_goals[i]))
        goals += [goals_padded] * l
        
    for sess in batch_usrdas:
        origin_responses_length += [len(sen) for sen in sess]
    max_response_length = max(origin_responses_length)
    for sess in batch_usrdas:
        origin_responses += padding(sess, max_response_length)
        
    for sess in batch_sysdas:
        sen_padded = padding(sess, 15)
        for j, sen in enumerate(sess):           
            if j == 0:
                post_single = np.zeros([max_sentence_num, 15], np.int)
                post_length_single = np.zeros([max_sentence_num], np.int)
            else:
                post_single = posts[-1]
                post_length_single = posts_length[-1]
            post_length_single[j] = min(len(sen), 15)
            post_single[j, :] = sen_padded[j]
            
            posts_length.append(post_length_single)
            posts.append(post_single)
    ''' end padding '''

    batch_input['origin_responses'] = origin_responses
    batch_input['origin_responses_length'] = origin_responses_length
    batch_input['posts_length'] = posts_length
    batch_input['posts'] = posts
    batch_input['goals_length'] = goals_length
    batch_input['goals'] = goals
    return model.train(batch_input)


def test(batch_goals, batch_usrdas, batch_sysdas):
    batch_input = {}
    posts_length = []
    posts = []
    origin_responses = []
    origin_responses_length = []
    goals_length = []
    goals = []

    ''' start padding '''
    sentence_num = [len(sess) for sess in batch_usrdas]
    max_sentence_num = max(sentence_num)

    max_goal_length = max([len(sess_goal) for sess_goal in batch_goals])
    for i, l in enumerate(sentence_num):
        goals_length += [len(batch_goals[i])] * l
        goals_padded = batch_goals[i] + [0] * (max_goal_length - len(batch_goals[i]))
        goals += [goals_padded] * l

    for sess in batch_usrdas:
        origin_responses_length += [len(sen) for sen in sess]
    max_response_length = max(origin_responses_length)
    for sess in batch_usrdas:
        origin_responses += padding(sess, max_response_length)

    for sess in batch_sysdas:
        sen_padded = padding(sess, 15)
        for j, sen in enumerate(sess):
            if j == 0:
                post_single = np.zeros([max_sentence_num, 15], np.int)
                post_length_single = np.zeros([max_sentence_num], np.int)
            else:
                post_single = posts[-1]
                post_length_single = posts_length[-1]
            post_length_single[j] = min(len(sen), 15)
            post_single[j, :] = sen_padded[j]

            posts_length.append(post_length_single)
            posts.append(post_single)
    ''' end padding '''

    batch_input['origin_responses'] = origin_responses
    batch_input['origin_responses_length'] = origin_responses_length
    batch_input['posts_length'] = posts_length
    batch_input['posts'] = posts
    batch_input['goals_length'] = goals_length
    batch_input['goals'] = goals
    return model.evaluate(batch_input)

def infer(batch_goals, batch_usrdas, batch_sysdas):
    batch_input = {}
    posts_length = []
    posts = []
    origin_responses = []
    origin_responses_length = []
    goals_length = []
    goals = []

    ''' start padding '''
    sentence_num = [len(sess) for sess in batch_usrdas]
    max_sentence_num = max(sentence_num)

    max_goal_length = max([len(sess_goal) for sess_goal in batch_goals])
    for i, l in enumerate(sentence_num):
        goals_length += [len(batch_goals[i])] * l
        goals_padded = batch_goals[i] + [0] * (max_goal_length - len(batch_goals[i]))
        goals += [goals_padded] * l

    for sess in batch_usrdas:
        origin_responses_length += [len(sen) for sen in sess]
    max_response_length = max(origin_responses_length)
    for sess in batch_usrdas:
        origin_responses += padding(sess, max_response_length)

    for sess in batch_sysdas:
        sen_padded = padding(sess, 15)
        for j, sen in enumerate(sess):
            if j == 0:
                post_single = np.zeros([max_sentence_num, 15], np.int)
                post_length_single = np.zeros([max_sentence_num], np.int)
            else:
                post_single = posts[-1]
                post_length_single = posts_length[-1]
            post_length_single[j] = min(len(sen), 15)
            post_single[j, :] = sen_padded[j]

            posts_length.append(post_length_single)
            posts.append(post_single)
    ''' end padding '''

    batch_input['origin_responses'] = origin_responses
    batch_input['origin_responses_length'] = origin_responses_length
    batch_input['posts_length'] = posts_length
    batch_input['posts'] = posts
    batch_input['goals_length'] = goals_length
    batch_input['goals'] = goals
    return model.infer(batch_input)

def get_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda x: x.lower() == 'true')
    parser.add_argument("--train", type="bool", default=True)
    parser.add_argument("--save_dir", type=str, default='save')
    args = parser.parse_args(sys.argv[1:])
    return args

def main():
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    idx = 0
    epoch_num = 10
    best = float('inf')

    if args.train:
        for i in range(epoch_num):
            generator = batch_iter(train_goals, train_usrdas, train_sysdas, BATCH_SIZE)
            print("Epoch {}".format(i))
            for batch_goals, batch_usrdas, batch_sysdas in generator:
                loss = train(batch_goals, batch_usrdas, batch_sysdas)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {}".format(time_str, idx, loss))
                idx += 1
                if idx % 100 == 0:
                    model.store_checkpoint(args.save_dir + '/' + str(i), 'latest')
                    val_loss, val_ppl, blue = test(val_goals, val_usrdas, val_sysdas)
                    print("Validate:")
                    print("{}:Val loss {} Val ppl {}".format(time_str, val_loss, val_ppl))
                    if val_loss < best:
                        best = val_loss
                        model.store_checkpoint(args.save_dir + '/' + str(i), 'best')
                    test_loss, test_ppl, bleu = test(test_goals, test_usrdas, test_sysdas)
                    rand = []

                    print("Test:")
                    print("{}:Test loss {} Test ppl {}".format(time_str, test_loss, test_ppl))
    else:
        _, perplexity, bleu, result = infer(batch_goals, batch_usrdas, batch_sysdas)
        print("ppl: {} BLEU: {}".format(perplexity, bleu))



if __name__ == "__main__":
    main()
