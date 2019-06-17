#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os

import numpy as np


def main():
    """This task consists of an MTurk agent evaluating a chit-chat model. They
    are asked to chat to the model adopting a specific persona. After their
    conversation, they are asked to evaluate their partner on several metrics.
    """
    parser = argparse.ArgumentParser(description='Analyze MEDkit experiment results')
    parser.add_argument(
        '-dp', '--datapath', default='./',
        help='path to datasets, defaults to current directory')

    args = parser.parse_args()

    dirs = os.listdir(args.datapath)

    num_s_dials = 0
    num_f_dials = 0
    dial_lens = []
    usr_turn_lens = []
    sys_turn_lens = []
    u_scores = []
    a_scores = []
    num_domains = []
    num_s_per_level = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    num_f_per_level = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for file in dirs:
        if os.path.isfile(os.path.join(args.datapath, file)) and file.endswith('json'):
            # print('open', os.path.join(args.datapath, file))
            with open(os.path.join(args.datapath, file)) as f:
                print(file)
                result = json.load(f)
                # pprint(result)
                level = len(result['goal']['domain_ordering'])
                if level > 5:
                    level = 5
                num_domains.append(level)
                if result['success']:
                    num_s_dials += 1
                    num_s_per_level[level] += 1
                else:
                    num_f_dials += 1
                    num_f_per_level[level] += 1
                dial_lens.append(len(result['dialog']))
                usr_lens = []
                sys_lens = []
                for i, turn in enumerate(result['dialog']):
                    if turn[0] == 0:
                        usr_turn_lens.append(len(turn[1].split()))
                    elif turn[0] == 1:
                        sys_turn_lens.append(len(turn[1].split()))
                u_scores.append(result['understanding_score'])
                a_scores.append(result['appropriateness_score'])
    print('Total number of dialogs:', num_s_dials + num_f_dials)
    print('Success rate:', num_s_dials/(num_s_dials + num_f_dials))
    for level in num_s_per_level:
        s_rate = 0 if num_s_per_level[level] + num_f_per_level[level] == 0 else\
             num_s_per_level[level] / (num_s_per_level[level] + num_f_per_level[level])
        print('Level {} success rate: {}'.format(level, s_rate))
    print('Avg dialog length: {}(+-{})'.format(np.mean(dial_lens), np.std(dial_lens)))
    print('Avg turn length: {}(+-{})'.format(np.mean(usr_turn_lens+sys_turn_lens), np.std(usr_turn_lens+sys_turn_lens)))
    print('Avg user turn length: {}(+-{})'.format(np.mean(usr_turn_lens), np.std(usr_turn_lens)))
    print('Avg system turn length: {}(+-{})'.format(np.mean(sys_turn_lens), np.std(sys_turn_lens)))
    print('Avg number of domains: {}(+-{})'.format(np.mean(num_domains), np.std(num_domains)))
    print('Avg understanding score: {}(+-{})'.format(np.mean(u_scores), np.std(u_scores)))
    print('Avg appropriateness score: {}(+-{})'.format(np.mean(a_scores), np.std(a_scores)))


if __name__ == '__main__':
    main()