# Modified by Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-

import logging
import os
import time

from user import UserNeural


def init_logging_handler(log_dir, extra=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(log_dir, current_time+extra))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

init_logging_handler('log', '_vhus')
env = UserNeural(True)

logging.debug('start training')

best = float('inf')
for e in range(10):
    env.imitating(e)
    best = env.imit_test(e, best)
