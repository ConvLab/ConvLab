# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import time
from queue import PriorityQueue
from threading import Thread

import tensorflow as tf
from convlab.modules.dst.multiwoz.mdbt import MDBTTracker, init_state
from convlab.modules.word_policy.multiwoz.mdrg.predict import loadModel, predict
from flask import Flask, request, jsonify

import convlab

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
_config = tf.ConfigProto()
_config.gpu_options.allow_growth = True
_config.allow_soft_placement = True
start_time = time.time()

rgi_queue = PriorityQueue(maxsize=0)
rgo_queue = PriorityQueue(maxsize=0)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def process():
    try:
        in_request = request.json
    except:
        return "invalid input: {}".format(in_request)
    rgi_queue.put(in_request)
    rgi_queue.join()
    output = rgo_queue.get()
    rgo_queue.task_done()
    return jsonify(output)


def generate_response(in_queue, out_queue):
    # Response generator
    response_model = loadModel(15)

    # state tracker
    sess = tf.Session(config=_config)
    mdbt = MDBTTracker()
    saver = tf.train.Saver()
    print('\tMDBT: model build time: {:.2f} seconds'.format(time.time() - start_time))
    mdbt.restore_model(sess, saver)
    prefix = os.path.dirname(convlab.__file__)
    dic = pickle.load(open(prefix + '/../data/nrg/mdrg/svdic.pkl', 'rb'))

    while True:
        # pop input
        in_request = in_queue.get()
        history = in_request['history']
        prev_state = in_request['prev_state']
        prev_active_domain = in_request['prev_active_domain']
        if prev_state is None:
            prev_state = init_state()
        state = init_state()
        state['history'] = history
        try:
            mdbt.state = state
            state = mdbt.update(sess, "")
        except Exception as e:
            print('State update error', e)
            prev_state = init_state()
            prev_active_domain = None
            state = init_state()
            history = [['null', 'hello']]
            state['history'] = history
        try:
            response, active_domain = predict(response_model, prev_state, prev_active_domain, state, dic)
        except Exception as e:
            print('Response generation error', e)
            response = 'What did you say?'
            active_domain = 'null'
        # print(response)
        out_queue.put({'response': response, 'active_domain': active_domain, 'state': state})
        in_queue.task_done()
        out_queue.join()


if __name__ == '__main__':
    worker = Thread(target=generate_response, args=(rgi_queue, rgo_queue,))
    worker.setDaemon(True)
    worker.start()

    app.run(host='0.0.0.0', port=10002)
