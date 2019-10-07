# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import math
import os
import sys
import time
from random import shuffle

import numpy as np
import tensorflow as tf

from convlab.modules.dst.multiwoz.dst_util import init_state, init_belief_state, normalize_value
from convlab.modules.dst.state_tracker import Tracker
from convlab.modules.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
from convlab.modules.word_dst.multiwoz.mdbt.mdbt_util import model_definition, load_word_vectors, load_ontology, \
    load_woz_data, \
    track_dialogue, generate_batch, process_history, evaluate_model

# DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), 'data/mdbt')
# VALIDATION_URL = os.path.join(DATA_PATH, "data/validate.json")
# WORD_VECTORS_URL = os.path.join(DATA_PATH, "word-vectors/paragram_300_sl999.txt")
# TRAINING_URL = os.path.join(DATA_PATH, "data/train.json")
# ONTOLOGY_URL = os.path.join(DATA_PATH, "data/ontology.json")
# TESTING_URL = os.path.join(DATA_PATH, "data/test.json")
# MODEL_URL = os.path.join(DATA_PATH, "models/model-1")
# GRAPH_URL = os.path.join(DATA_PATH, "graphs/graph-1")
# RESULTS_URL = os.path.join(DATA_PATH, "results/log-1.txt")
# KB_URL = os.path.join(DATA_PATH, "data/")  # TODO: yaoqin
# TRAIN_MODEL_URL = os.path.join(DATA_PATH, "train_models/model-1")
# TRAIN_GRAPH_URL = os.path.join(DATA_PATH, "train_graph/graph-1")

train_batch_size = 1
batches_per_eval = 10
no_epochs = 600
device = "gpu"
start_batch = 0


class MDBTTracker(Tracker):
    """
    A multi-domain belief tracker, adopted from https://github.com/osmanio2/multi-domain-belief-tracking.
    """
    def __init__(self, data_dir='data/mdbt'):
        Tracker.__init__(self)
        # data profile
        self.data_dir = data_dir
        self.validation_url = os.path.join(self.data_dir, 'data/validate.json')
        self.word_vectors_url = os.path.join(self.data_dir, 'word-vectors/paragram_300_sl999.txt')
        self.training_url = os.path.join(self.data_dir, 'data/train.json')
        self.ontology_url = os.path.join(self.data_dir, 'data/ontology.json')
        self.testing_url = os.path.join(self.data_dir, 'data/test.json')
        self.model_url = os.path.join(self.data_dir, 'models/model-1')
        self.graph_url = os.path.join(self.data_dir, 'graphs/graph-1')
        self.results_url = os.path.join(self.data_dir, 'results/log-1.txt')
        self.kb_url = os.path.join(self.data_dir, 'data/')  # not used
        self.train_model_url = os.path.join(self.data_dir, 'train_models/model-1')
        self.train_graph_url = os.path.join(self.data_dir, 'train_graph/graph-1')

        print('Configuring MDBT model...')
        self.word_vectors = load_word_vectors(self.word_vectors_url)

        # Load the ontology and extract the feature vectors
        self.ontology, self.ontology_vectors, self.slots = load_ontology(self.ontology_url, self.word_vectors)

        # Load and process the training data
        self.dialogues, self.actual_dialogues = load_woz_data(self.testing_url, self.word_vectors, self.ontology)
        self.no_dialogues = len(self.dialogues)

        self.model_variables = model_definition(self.ontology_vectors, len(self.ontology), self.slots, num_hidden=None,
                                                bidir=True, net_type=None, test=True, dev='cpu')
        self.state = init_state()
        _config = tf.ConfigProto()
        _config.gpu_options.allow_growth = True
        _config.allow_soft_placement = True
        self.sess = tf.Session(config=_config)
        self.param_restored = False
        self.det_dic = {}
        for domain, dic in REF_USR_DA.items():
            for key, value in dic.items():
                assert '-' not in key
                self.det_dic[key.lower()] = key + '-' + domain
                self.det_dic[value.lower()] = key + '-' + domain
        self.value_dict = json.load(open(os.path.join(self.data_dir, '../multiwoz/value_dict.json')))

    def init_session(self):
        self.state = init_state()
        if not self.param_restored:
            self.restore()

    def restore(self):
        self.restore_model(self.sess, tf.train.Saver())

    def update(self, user_act=None):
        """Update the dialog state."""
        if not isinstance(user_act, str):
            raise Exception('Expected user_act to be <class \'str\'> type, but get {}.'.format(type(user_act)))
        prev_state = self.state
        if not os.path.exists(os.path.join(self.data_dir, "results")):
            os.makedirs(os.path.join(self.data_dir, "results"))

        global train_batch_size

        model_variables = self.model_variables
        (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
         slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, predictions,
         true_predictions, [y, _]) = model_variables

        # generate fake dialogue based on history (this os to reuse the original MDBT code)
        # actual_history = prev_state['history']  # [[sys, user], [sys, user], ...]
        actual_history = copy.deepcopy(prev_state['history'])  # [[sys, user], [sys, user], ...]
        actual_history[-1].append(user_act)
        actual_history = self.normalize_history(actual_history)
        if len(actual_history) == 0:
            actual_history = [['', user_act if len(user_act)>0 else 'fake user act']]
        fake_dialogue = {}
        turn_no = 0
        for _sys, _user in actual_history:
            turn = {}
            turn['system'] = _sys
            fake_user = {}
            fake_user['text'] = _user
            fake_user['belief_state'] = init_belief_state
            turn['user'] = fake_user
            key = str(turn_no)
            fake_dialogue[key] = turn
            turn_no += 1
        context, actual_context = process_history([fake_dialogue], self.word_vectors, self.ontology)
        batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len, \
                batch_no_turns = generate_batch(context, 0, 1, len(self.ontology))  # old feature

        # run model
        [pred, y_pred] = self.sess.run(
            [predictions, y],
            feed_dict={user: batch_user, sys_res: batch_sys,
                       labels: batch_labels,
                       domain_labels: batch_domain_labels,
                       user_uttr_len: batch_user_uttr_len,
                       sys_uttr_len: batch_sys_uttr_len,
                       no_turns: batch_no_turns,
                       keep_prob: 1.0})

        # convert to str output
        dialgs, _, _ = track_dialogue(actual_context, self.ontology, pred, y_pred)
        assert len(dialgs) >= 1
        last_turn = dialgs[0][-1]
        predictions = last_turn['prediction']
        new_belief_state = copy.deepcopy(prev_state['belief_state'])

        # updaet belief state
        for item in predictions:
            item = item.lower()
            domain, slot, value = item.strip().split('-')
            value = value[::-1].split(':', 1)[1][::-1]
            if slot == 'price range':
                slot = 'pricerange'
            if slot not in ['name', 'book']:
                if domain not in new_belief_state:
                    raise Exception('Error: domain <{}> not in belief state'.format(domain))
                slot = REF_SYS_DA[domain.capitalize( )].get(slot, slot)
                assert 'semi' in new_belief_state[domain]
                assert 'book' in new_belief_state[domain]
                if 'book' in slot:
                    assert slot.startswith('book ')
                    slot = slot.strip().split()[1]
                domain_dic = new_belief_state[domain]
                if slot in domain_dic['semi']:
                    new_belief_state[domain]['semi'][slot] = normalize_value(self.value_dict, domain, slot, value)
                elif slot in domain_dic['book']:
                    new_belief_state[domain]['book'][slot] = value
                elif slot.lower() in domain_dic['book']:
                    new_belief_state[domain]['book'][slot.lower()] = value
                else:
                    with open('mdbt_unknown_slot.log', 'a+') as f:
                        f.write('unknown slot name <{}> with value <{}> of domain <{}>\nitem: {}\n\n'.format(slot, value,
                                domain, item))
        new_request_state = copy.deepcopy(prev_state['request_state'])
        # update request_state
        user_request_slot = self.detect_requestable_slots(user_act)
        for domain in user_request_slot:
            for key in user_request_slot[domain]:
                if domain not in new_request_state:
                    new_request_state[domain] = {}
                if key not in new_request_state[domain]:
                    new_request_state[domain][key] = user_request_slot[domain][key]
        # update state
        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        new_state['request_state'] = new_request_state
        self.state = new_state
        return self.state

    def normalize_history(self, history):
        """Replace zero-length history."""
        for i in range(len(history)):
            a, b = history[i]
            if len(a) == 0:
                history[i][0] = 'sys'
            if len(b) == 0:
                history[i][1] = 'user'
        return history

    def detect_requestable_slots(self, observation):
        result = {}
        observation = observation.lower()
        _observation = ' {} '.format(observation)
        for value in self.det_dic.keys():
            _value = ' {} '.format(value.strip())
            if _value in _observation:
                key, domain = self.det_dic[value].split('-')
                if domain not in result:
                    result[domain] = {}
                result[domain][key] = 0
        return result

    def restore_model(self, sess, saver):
        saver.restore(sess, self.model_url)
        print('Loading trained MDBT model from ', self.model_url)
        self.param_restored = True

    def train(self):
        """
            Train the model.
            Model saved to
        """
        num_hid, bidir, net_type, n2p, batch_size, model_url, graph_url, dev = \
                None, True, None, None, None, None, None, None
        global train_batch_size, MODEL_URL, GRAPH_URL, device, TRAIN_MODEL_URL, TRAIN_GRAPH_URL

        if batch_size:
            train_batch_size = batch_size
            print("Setting up the batch size to {}.........................".format(batch_size))
        if model_url:
            TRAIN_MODEL_URL = model_url
            print("Setting up the model url to {}.........................".format(TRAIN_MODEL_URL))
        if graph_url:
            TRAIN_GRAPH_URL = graph_url
            print("Setting up the graph url to {}.........................".format(TRAIN_GRAPH_URL))

        if dev:
            device = dev
            print("Setting up the device to {}.........................".format(device))

        # 1 Load and process the input data including the ontology
        # Load the word embeddings
        word_vectors = load_word_vectors(self.word_vectors_url)

        # Load the ontology and extract the feature vectors
        ontology, ontology_vectors, slots = load_ontology(self.ontology_url, word_vectors)

        # Load and process the training data
        dialogues, _ = load_woz_data(self.training_url, word_vectors, ontology)
        no_dialogues = len(dialogues)

        # Load and process the validation data
        val_dialogues, _ = load_woz_data(self.validation_url, word_vectors, ontology)

        # Generate the validation batch data
        val_data = generate_batch(val_dialogues, 0, len(val_dialogues), len(ontology))
        val_iterations = int(len(val_dialogues) / train_batch_size)

        # 2 Initialise and set up the model graph
        # Initialise the model
        graph = tf.Graph()
        with graph.as_default():
            model_variables = model_definition(ontology_vectors, len(ontology), slots, num_hidden=num_hid, bidir=bidir,
                                               net_type=net_type, dev=device)
            (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
             slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, _, _, _) = model_variables
            [precision, recall, value_f1] = value_f1
            saver = tf.train.Saver()
            if device == 'gpu':
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto(device_count={'GPU': 0})

            sess = tf.Session(config=config)
            if os.path.exists(TRAIN_MODEL_URL + ".index"):
                saver.restore(sess, TRAIN_MODEL_URL)
                print("Loading from an existing model {} ....................".format(TRAIN_MODEL_URL))
            else:
                if not os.path.exists(TRAIN_MODEL_URL):
                    os.makedirs('/'.join(TRAIN_MODEL_URL.split('/')[:-1]))
                    os.makedirs('/'.join(TRAIN_GRAPH_URL.split('/')[:-1]))
                init = tf.global_variables_initializer()
                sess.run(init)
                print("Create new model parameters.....................................")
            merged = tf.summary.merge_all()
            val_accuracy = tf.summary.scalar('validation_accuracy', value_accuracy)
            val_f1 = tf.summary.scalar('validation_f1_score', value_f1)
            train_writer = tf.summary.FileWriter(TRAIN_GRAPH_URL, graph)
            train_writer.flush()

        # 3 Perform an epoch of training
        last_update = -1
        best_f_score = -1
        for epoch in range(no_epochs):

            batch_size = train_batch_size
            sys.stdout.flush()
            iterations = math.ceil(no_dialogues / train_batch_size)
            start_time = time.time()
            val_i = 0
            shuffle(dialogues)
            for batch_id in range(iterations):

                if batch_id == iterations - 1 and no_dialogues % iterations != 0:
                    batch_size = no_dialogues % train_batch_size

                batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len, \
                batch_no_turns = generate_batch(dialogues, batch_id, batch_size, len(ontology))

                [_, summary, da, sa, va, vf, pr, re] = sess.run([train_step, merged, domain_accuracy, slot_accuracy,
                                                                 value_accuracy, value_f1, precision, recall],
                                                                feed_dict={user: batch_user, sys_res: batch_sys,
                                                                           labels: batch_labels,
                                                                           domain_labels: batch_domain_labels,
                                                                           user_uttr_len: batch_user_uttr_len,
                                                                           sys_uttr_len: batch_sys_uttr_len,
                                                                           no_turns: batch_no_turns,
                                                                           keep_prob: 0.5})

                print("The accuracies for domain is {:.2f}, slot {:.2f}, value {:.2f}, f1_score {:.2f} precision {:.2f}"
                      " recall {:.2f} for batch {}".format(da, sa, va, vf, pr, re, batch_id + iterations * epoch))

                train_writer.add_summary(summary, start_batch + batch_id + iterations * epoch)

                # ================================ VALIDATION ==============================================

                if batch_id % batches_per_eval == 0 or batch_id == 0:
                    if batch_id == 0:
                        print("Batch", "0", "to", batch_id, "took", round(time.time() - start_time, 2), "seconds.")

                    else:
                        print("Batch", batch_id + iterations * epoch - batches_per_eval, "to",
                              batch_id + iterations * epoch, "took",
                              round(time.time() - start_time, 3), "seconds.")
                        start_time = time.time()

                    _, _, v_acc, f1_score, sm1, sm2 = evaluate_model(sess, model_variables, val_data,
                                                                     [val_accuracy, val_f1], batch_id, val_i)
                    val_i += 1
                    val_i %= val_iterations
                    train_writer.add_summary(sm1, start_batch + batch_id + iterations * epoch)
                    train_writer.add_summary(sm2, start_batch + batch_id + iterations * epoch)
                    stime = time.time()
                    current_metric = f1_score
                    print(" Validation metric:", round(current_metric, 5), " eval took",
                          round(time.time() - stime, 2), "last update at:", last_update, "/", iterations)

                    # and if we got a new high score for validation f-score, we need to save the parameters:
                    if current_metric > best_f_score:
                        last_update = batch_id + iterations * epoch + 1
                        print("\n ====================== New best validation metric:", round(current_metric, 4),
                              " - saving these parameters. Batch is:", last_update, "/", iterations,
                              "---------------- ===========  \n")

                        best_f_score = current_metric

                        saver.save(sess, TRAIN_MODEL_URL)

            print("The best parameters achieved a validation metric of", round(best_f_score, 4))

    def test(self, sess):
        """Test the MDBT model on mdbt dataset. Almost the same as original code."""
        if not os.path.exists("../../data/mdbt/results"):
            os.makedirs("../../data/mdbt/results")

        global train_batch_size, MODEL_URL, GRAPH_URL

        model_variables = self.model_variables
        (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
         slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, predictions,
         true_predictions, [y, _]) = model_variables
        [precision, recall, value_f1] = value_f1
        # print("\tMDBT: Loading from an existing model {} ....................".format(MODEL_URL))

        iterations = math.ceil(self.no_dialogues / train_batch_size)
        batch_size = train_batch_size
        [slot_acc, tot_accuracy] = [np.zeros(len(self.ontology), dtype="float32"), 0]
        slot_accurac = 0
        # value_accurac = np.zeros((len(slots),), dtype="float32")
        value_accurac = 0
        joint_accuracy = 0
        f1_score = 0
        preci = 0
        recal = 0
        processed_dialogues = []
        # np.set_printoptions(threshold=np.nan)
        for batch_id in range(int(iterations)):

            if batch_id == iterations - 1:
                batch_size = self.no_dialogues - batch_id * train_batch_size

            batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len, \
            batch_no_turns = generate_batch(self.dialogues, batch_id, batch_size, len(self.ontology))

            [da, sa, va, vf, pr, re, pred, true_pred, y_pred] = sess.run(
                [domain_accuracy, slot_accuracy, value_accuracy,
                 value_f1, precision, recall, predictions,
                 true_predictions, y],
                feed_dict={user: batch_user, sys_res: batch_sys,
                           labels: batch_labels,
                           domain_labels: batch_domain_labels,
                           user_uttr_len: batch_user_uttr_len,
                           sys_uttr_len: batch_sys_uttr_len,
                           no_turns: batch_no_turns,
                           keep_prob: 1.0})

            true = sum([1 if np.array_equal(pred[k, :], true_pred[k, :]) and sum(true_pred[k, :]) > 0 else 0
                        for k in range(true_pred.shape[0])])
            actual = sum([1 if sum(true_pred[k, :]) > 0 else 0 for k in range(true_pred.shape[0])])
            ja = true / actual
            tot_accuracy += da
            # joint_accuracy += ja
            slot_accurac += sa
            if math.isnan(pr):
                pr = 0
            preci += pr
            recal += re
            if math.isnan(vf):
                vf = 0
            f1_score += vf
            # value_accurac += va
            slot_acc += np.mean(np.asarray(np.equal(pred, true_pred), dtype="float32"), axis=0)

            dialgs, va1, ja = track_dialogue(self.actual_dialogues[batch_id * train_batch_size:
                                             batch_id * train_batch_size + batch_size],
                                             self.ontology, pred, y_pred)
            processed_dialogues += dialgs
            joint_accuracy += ja
            value_accurac += va1

            print(
                "The accuracies for domain is {:.2f}, slot {:.2f}, value {:.2f}, other value {:.2f}, f1_score {:.2f} precision {:.2f}"
                " recall {:.2f}  for batch {}".format(da, sa, np.mean(va), va1, vf, pr, re, batch_id))

        print(
            "End of evaluating the test set...........................................................................")

        slot_acc /= iterations
        # print("The accuracies for each slot:")
        # print(value_accurac/iterations)
        print("The overall accuracies for domain is"
              " {}, slot {}, value {}, f1_score {}, precision {},"
              " recall {}, joint accuracy {}".format(tot_accuracy / iterations, slot_accurac / iterations,
                                                     value_accurac / iterations, f1_score / iterations,
                                                     preci / iterations, recal / iterations,
                                                     joint_accuracy / iterations))

        with open(self.results_url, 'w') as f:
            json.dump(processed_dialogues, f, indent=4)


def test_update():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True
    _config.allow_soft_placement = True
    start_time = time.time()
    mdbt = MDBTTracker()
    print('\tMDBT: model build time: {:.2f} seconds'.format(time.time() - start_time))
    saver = tf.train.Saver()
    mdbt.restore_model(mdbt.sess, saver)
    # demo state history
    mdbt.state['history'] = [['null', 'I\'m trying to find an expensive restaurant in the centre part of town.'],
                             [
                                 'The Cambridge Chop House is an good expensive restaurant in the centre of town. Would you like me to book it for you?',
                                 'Yes, a table for 1 at 16:15 on sunday.  I need the reference number.']]
    new_state = mdbt.update(None, 'hi, this is not good')
    print(json.dumps(new_state, indent=4))
    print('all time: {:.2f} seconds'.format(time.time() - start_time))


def evaluate_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True
    _config.allow_soft_placement = True
    start_time = time.time()
    mdbt = MDBTTracker()
    print('\tMDBT: model build time: {:.2f} seconds'.format(time.time() - start_time))
    saver = tf.train.Saver()
    mdbt.restore_model(mdbt.sess, saver)
    mdbt.test(mdbt.sess)

if __name__ == '__main__':
    evaluate_model()
