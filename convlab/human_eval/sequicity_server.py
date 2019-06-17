# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pprint import pprint
from queue import PriorityQueue
from threading import Thread

from flask import Flask, request, jsonify

from convlab.modules.e2e.multiwoz.Sequicity.model import main as sequicity_load

rgi_queue = PriorityQueue(maxsize=0)
rgo_queue = PriorityQueue(maxsize=0)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def process():
    try:
        in_request = request.json
        print(in_request)
    except:
        return "invalid input: {}".format(in_request)
    rgi_queue.put(in_request)
    rgi_queue.join()
    output = rgo_queue.get()
    print(output['response'])
    rgo_queue.task_done()
    return jsonify(output)


def generate_response(in_queue, out_queue):
    # Load Sequicity model
    sequicity = sequicity_load('load', 'tsdf-multiwoz')

    while True:
        # pop input
        in_request = in_queue.get()
        state = in_request['state']
        input = in_request['input']
        pprint(in_request)
        try:
            state = sequicity.predict(input, state)
        except Exception as e:
            print('State update error', e)
            state = {}
        pprint(state)
        try:
            response = state['sys']
        except Exception as e:
            print('Response generation error', e)
            response = 'What did you say?'
        out_queue.put({'response': response, 'state': state})
        in_queue.task_done()
        out_queue.join()


if __name__ == '__main__':
    worker = Thread(target=generate_response, args=(rgi_queue, rgo_queue,))
    worker.setDaemon(True)
    worker.start()

    app.run(host='0.0.0.0', port=10001)
