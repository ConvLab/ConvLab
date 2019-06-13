# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from pprint import pprint
from queue import PriorityQueue
from threading import Thread

from convlab.modules.nlu.multiwoz.mlst.nlu import MlstNLU
from flask import Flask, request, jsonify

from convlab import RuleBasedMultiwozBot
from convlab.modules.dst.multiwoz.rule_dst import RuleDST
from convlab.modules.nlg.multiwoz.multiwoz_template_nlg.multiwoz_template_nlg import MultiwozTemplateNLG

rgi_queue = PriorityQueue(maxsize=0)
rgo_queue = PriorityQueue(maxsize=0)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def process():
    try:
        in_request = request.json
        print('start process')
        print(in_request)
    except:
        return "invalid input: {}".format(in_request)
    rgi_queue.put(in_request)
    rgi_queue.join()
    output = rgo_queue.get()
    print(output['response'])
    rgo_queue.task_done()
    # return jsonify({'response': response}) 
    return jsonify(output)


def generate_response(in_queue, out_queue):
    # Load Sequicity model
    rulebot = RuleBasedMultiwozBot()
    nlu = MlstNLU()
    tracker = RuleDST()
    agent_nlg = MultiwozTemplateNLG(is_user=False)

    while True:
        # pop input
        in_request = in_queue.get()

        state = in_request['state']
        recommend_flag = in_request['recommend_flag']
        user_input = in_request['input']
        rulebot.last_state = copy.deepcopy(state)
        rulebot.recommend_flag = recommend_flag

        if not state == {}:
            tracker.state = copy.deepcopy(state)

        useract = nlu.parse(user_input)
        pprint(useract)
        tracker.update(user_act=useract)
        tracker.state['user_action'] = useract
        state = copy.deepcopy(tracker.state)

        pprint(in_request)
        try:
            agent_diaact = rulebot.predict(state)
            agent_NL = agent_nlg.generate(agent_diaact)
            recommend_flag = rulebot.recommend_flag
            state['history'].append([agent_diaact, useract])
        except Exception as e:
            print('State update error', e)
            state = {}
            recommend_flag = -1
        # pprint(state)

        try:
            response = agent_NL
            if response == '':
                response = 'Sorry I donot understand, can you paraphrase?'
        except Exception as e:
            print('Response generation error', e)
            response = 'What did you say?'
        # print(response)
        out_queue.put({'response': response, 'state': copy.deepcopy(state), 'recommend_flag': recommend_flag})
        in_queue.task_done()
        out_queue.join()


if __name__ == '__main__':
    worker = Thread(target=generate_response, args=(rgi_queue, rgo_queue,))
    worker.setDaemon(True)
    worker.start()

    app.run(host='0.0.0.0', port=10003)

    # import random
    # random.seed(100)
    #
    # user_input = "I would like to have a hotel in the place of west"
    # rulebot = RuleBasedMultiwozBot()
    # nlu = MlstNLU()
    # tracker = RuleDST()
    # user_nlg = MultiwozTemplateNLG(is_user=False)
    # useract = nlu.parse(user_input)
    # tracker.state['user_action'] = useract
    # pprint(useract)
    # pprint('---------------------------------------------')
    # state = tracker.update(user_act=useract)
    # # pprint(state)
    # agent_diaact = rulebot.predict(state)
    # pprint('Agent action')
    # pprint(agent_diaact)
    # pprint('---------------------------------------------')
    # agent_NL = user_nlg.generate(agent_diaact)
    # pprint(agent_NL)
    # pprint('---------------------------------------------')
    # tracker.state['history'].append([agent_diaact, useract])
    # state = copy.deepcopy(tracker.state)
    #
    # rulebot = RuleBasedMultiwozBot()
    # nlu = MlstNLU()
    # tracker = RuleDST()
    # tracker.state = copy.deepcopy(state)
    # user_nlg = MultiwozTemplateNLG(is_user=False)
    # rulebot.last_state = copy.deepcopy(state)
    #
    # user_input = "I would like to have an expensive hotel in the place of east"
    # useract = nlu.parse(user_input)
    # tracker.state['user_action'] = useract
    # pprint(useract)
    # pprint('---------------------------------------------')
    # state = tracker.update(user_act=useract)
    # # pprint(state)
    # agent_diaact = rulebot.predict(state)
    # pprint('Agent action')
    # pprint(agent_diaact)
    # pprint('---------------------------------------------')
    # agent_NL = user_nlg.generate(agent_diaact)
    # pprint(agent_NL)
    # pprint('---------------------------------------------')
    # tracker.state['history'].append([agent_diaact, useract])
