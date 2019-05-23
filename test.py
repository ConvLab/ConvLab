# coding='utf-8'
import os
import tensorflow as tf
from convlab import *

# demo setting
params = dict()
params['session_num'] = 200
params['cuda_id'] = '0'

# TF setting
os.environ["CUDA_VISIBLE_DEVICES"] = params['cuda_id']
_config = tf.ConfigProto()
_config.gpu_options.allow_growth = True
_config.allow_soft_placement = True
global_sess = tf.Session(config=_config)

############ components for system bot ############

uni_nlu = SVMNLU()

sys_tracker = RuleDST()  # Rule DST

sys_policy = RuleBasedMultiwozBot()  # Rule Multiwoz Policy

sys_nlg = MultiwozTemplateNLG(is_user=False)  # template NLG

# aggregate system components
system_bot = DialogSystem(uni_nlu, sys_tracker, sys_policy, None)

############ components for user bot ############

user_policy = UserPolicyAgendaMultiWoz()  # Agenda-based Simulator (act-in act-out)

user_nlg = MultiwozTemplateNLG(is_user=True)  # template NLG

# aggregate user components
user_simulator = UserSimulator(None, user_policy, user_nlg)

# setup session controller
session_controller = Session(system_bot, user_simulator)
logger = Log('session.txt')
logger.clear()

stat = {'success': 0, 'fail': 0}

for session_id in range(params['session_num']):
    session_over = False
    last_user_response = user_simulator.init_response()
    session_controller.init_session()
    session_controller.sess = global_sess

    print('******Episode %d******' % (session_id))
    print(user_simulator.policy.goal)

    while not session_over:
        system_response, user_response, session_over, reward = session_controller.next_turn(last_user_response)
        if not session_over:
            last_user_response = user_response

        sys_da, user_da = session_controller.action_history[-1]  # action_history stores the actions of both agents

        # print('\tstate: {}'.format(system_bot.tracker.state.keys()))
        print('\tsystem user_da: ' + '{}'.format(system_bot.user_act))
        print('\tsystem da: {}'.format(sys_da))
        print('\tsystem: ' + '{}'.format(system_response))
        print('\t------------------------------------------------------')
        print('\tuser sys_da: ' + '{}'.format(user_simulator.sys_act))
        print('\tuser da: {}'.format(user_da))
        print('\tuser: ' + '{}'.format(user_response))
        print('\t--- turn end ---')

    dialog_status = user_simulator.policy.goal.task_complete()
    if dialog_status:
        stat['success'] += 1
    else:
        stat['fail'] += 1

    print('task completion: {}'.format(user_simulator.policy.goal.task_complete()))
    logger.log('---- session end ----')
    print('---- session end ----')
    # session_controller.train_sys()  # train the params of system agent

print('\nstatistics: %s' % (stat))