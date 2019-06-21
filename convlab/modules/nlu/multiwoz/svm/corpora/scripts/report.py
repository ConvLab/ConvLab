from __future__ import print_function
# Modified by Microsoft Corporation.
# Licensed under the MIT license.

###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2019
# Cambridge University Engineering Department Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

import argparse
import os
import sys

SCHEDULES = [1,2]
LABEL_SCHEMES = ["a","b"]
EVALUATION_SCHEMES = {
    (1,"a"): "eval_1a",
    (1,"b"): "eval_1b",
    (2,"a"): "eval_2a",
    (2,"b"): "eval_2b",
}

def main(argv):
    #
    # CMD LINE ARGS
    # 
    install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description='Formats a scorefile into a report and prints to stdout.')
    parser.add_argument('--scorefile',dest='csv',action='store',required=True,metavar='CSV_FILE',
                        help='File to read with CSV scoring data')
    args = parser.parse_args()

    csvfile = open(args.csv)
    #  "state_component, stat, schedule, label_scheme, N, result"
    header = True
    tables = {}
    for state_component in ["goal.joint","requested.all","method"]:
        tables[state_component] = {}
        for evaluation_scheme in EVALUATION_SCHEMES.values():
            tables[state_component][evaluation_scheme] = {}
        
    basic_stats = {}
    
    for line in open(args.csv):
        if header:
            header = False
            continue
        state_component, stat, schedule, label_scheme, N, result = line.split(",")
        stat = stat.strip()
        if state_component == "basic" :
            basic_stats[stat] = result.strip()
        else :
            N = int(N)
            schedule = int(schedule)
            label_scheme = (label_scheme).strip()
            result = result.strip()
            if result != "-" :
                result = "%.7f" % float(result)
            
            if state_component in tables.keys() :
                tables[state_component][EVALUATION_SCHEMES[(schedule, label_scheme)]][stat] = result
    
    for state_component in ["goal.joint","method","requested.all"]:
        print(state_component.center(50))
        evaluation_schemes = sorted([key for key in tables[state_component].keys() if len(tables[state_component][key])>0])
        stats = tables[state_component][evaluation_schemes[0]].keys()
        stats.sort()
        print_row(['']+evaluation_schemes, header=True)
        for stat in stats:
            print_row([stat] + [tables[state_component][evaluation_scheme][stat] for evaluation_scheme in evaluation_schemes])
        
        print("\n\n")
            
        
            
    
    print('                                    featured metrics')
    print_row(["","Joint Goals","Requested","Method"],header=True)
    print_row(["Accuracy",tables["goal.joint"]["eval_2a"]["acc"],tables["requested.all"]["eval_2a"]["acc"],tables["method"]["eval_2a"]["acc"] ])
    print_row(["l2",tables["goal.joint"]["eval_2a"]["l2"],tables["requested.all"]["eval_2a"]["l2"],tables["method"]["eval_2a"]["l2"] ])
    print_row(["roc.v2_ca05",tables["goal.joint"]["eval_2a"]["roc.v2_ca05"],tables["requested.all"]["eval_2a"]["roc.v2_ca05"],tables["method"]["eval_2a"]["roc.v2_ca05"] ])
    
    
    print("\n\n")
    
    
    print('                                    basic stats')
    print('-----------------------------------------------------------------------------------')
    for k in sorted(basic_stats.keys()):
        v = basic_stats[k]
        print('%20s : %s' % (k,v))

def print_row(row, header=False):
    out = [str(x) for x in row]
    for i in range(len(out)):
        if i==0 :
            out[i] = out[i].ljust(14)
        else :
            out[i] = out[i].center(17)
    
    out = ("|".join(out))[:-1]+"|"
    
    if header:
        print("-"*len(out))
        print(out)
        print("-"*len(out))
    else:
        print(out)


if (__name__ == '__main__'):
    main(sys.argv)

