# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys

sys.path.append('../../')
from convlab.agent import Body
from convlab.agent import DialogAgent
from convlab.spec import spec_util
from convlab.env import make_env

import numpy as np
import copy
from flask import Flask, request, jsonify
from queue import PriorityQueue
from threading import Thread
import time

rgi_queue = PriorityQueue(maxsize=0)
rgo_queue = PriorityQueue(maxsize=0)

app = Flask(__name__)

os.environ['lab_mode'] = 'eval'
spec_file = sys.argv[1]
spec_name = sys.argv[2]
lab_mode = sys.argv[3]

if '@' in lab_mode:
    lab_mode, prename = lab_mode.split('@')
    spec = spec_util.get_eval_spec(spec_file, spec_name, prename)
else:
    spec = spec_util.get(spec_file, spec_name)

# # lab_mode, prename = sys.argv[3].split('@')
# spec = spec_util.get_eval_spec(spec_file, prename)
spec = spec_util.override_eval_spec(spec)
agent_spec = spec['agent'][0]
env = make_env(spec)
body = Body(env, spec['agent'])
agent = DialogAgent(spec, body)

# last_obs = 'hi'
# agent.reset(last_obs)


# obs = 'hi can you find me a hotel in the west?'
# action = agent.act(obs)
# next_obs = 'we have six people'
# agent.update(obs, action, 0, next_obs, 0)


# action = agent.act(next_obs)

@app.route('/', methods=['GET', 'POST'])
def process():
    try:
        in_request = request.json
        print(in_request)
    except:
        return "invalid input: {}".format(in_request)
    rgi_queue.put((time.time(), in_request))
    rgi_queue.join()
    output = rgo_queue.get()
    print(output['response'])
    rgo_queue.task_done()
    # return jsonify({'response': response})
    return jsonify(output)


def generate_response(in_queue, out_queue):
    while True:
        # pop input
        last_action = 'null'
        in_request = in_queue.get()
        obs = in_request['input']

        if in_request['agent_state'] == {}:
            agent.reset(obs)
        else:
            encoded_state, dst_state, last_action = in_request['agent_state']
            agent.body.encoded_state = np.asarray(encoded_state) if isinstance(encoded_state, list) else encoded_state
            agent.dst.state = copy.deepcopy(dst_state)
            agent.update(obs, last_action, 0, obs, 0)
        try:
            action = agent.act(obs)
            encoded_state = agent.body.encoded_state.tolist() if isinstance(agent.body.encoded_state,
                                                                            np.ndarray) else agent.body.encoded_state
            dst_state = copy.deepcopy(agent.dst.state)
        except Exception as e:
            print('agent error', e)

        try:
            if action == '':
                response = 'Sorry I do not understand, can you paraphrase?'
            else:
                response = action
        except Exception as e:
            print('Response generation error', e)
            response = 'What did you say?'

        last_action = action
        out_queue.put({'response': response, 'agent_state': (encoded_state, dst_state, last_action)})
        in_queue.task_done()
        out_queue.join()


if __name__ == '__main__':
    worker = Thread(target=generate_response, args=(rgi_queue, rgo_queue,))
    worker.setDaemon(True)
    worker.start()

    app.run(host='0.0.0.0', port=10004)
