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
import json
import math
import os
import sys
from collections import defaultdict

import baseline

# type, and task are evaluated- shouldn't be.

eps = 0.001 # domain for math.log

def main(argv):
    #
    # CMD LINE ARGS
    #
    install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    utils_dirname = os.path.join(install_path,'lib')
    
    sys.path.append(utils_dirname)
    from dataset_walker import dataset_walker
    list_dir = os.path.join(install_path,'config')

    parser = argparse.ArgumentParser(description='Evaluate output from a belief tracker.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True,
                        help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True,
                        help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--decodefile',dest='decodefile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing decoder output JSON')
    parser.add_argument('--scorefile',dest='csv',action='store',metavar='CSV_FILE',required=True,
                        help='File to write with CSV scoring data')
    parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,
                        help='JSON Ontology file')
    parser.add_argument('--trackerfile',dest='trackerfile',action='store',metavar='JSON_FILE',required=True,
                        help='Tracker JSON file for output')
    
    
    args = parser.parse_args()

    sessions = dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)
    decode_results = json.load(open(args.decodefile))
    ontology = json.load(open(args.ontology))

    metrics = {
        "tophyp":Fscore(ontology),
        "ice":ICE(ontology)
    }
    
    belief_metrics = {
        "accuracy":BeliefAccuracy(ontology)
    }
    
    # we run the baseline focus tracker on the output of the SLU
    tracker = baseline.FocusTracker()
    tracker_output = {"sessions":[],"wall-time":0.0}
    tracker_output["dataset"]  = args.dataset
    
    for call, decode_session in zip(sessions, decode_results["sessions"]):
        tracker.reset()
        this_session = {"session-id":call.log["session-id"], "turns":[]}
        for (log_turn, label), decode_result in zip(call, decode_session["turns"]):
       
            true_label = label["semantics"]["json"]
            slu_hyps = decode_result["slu-hyps"]
            slu_hyps.sort(key=lambda x:-x["score"])
            total_p = sum([x["score"] for x in slu_hyps])
            if total_p > 1.0 :
                if total_p > 1.00001 :
                    print("Warning: total_p =",total_p,"> 1.0- renormalising.")
                for slu_hyp in slu_hyps:
                    slu_hyp["score"] = slu_hyp["score"]/total_p
            
            
            for metric in metrics.values():
                metric.add_turn(true_label, slu_hyps, log_turn, label)
                
            # for passing to tracker
            this_turn = {
                    "input":{"live":{"slu-hyps":slu_hyps}},
                    "output":log_turn["output"]
            }
            goal_hyps = tracker.addTurn(this_turn)
            for belief_metric in belief_metrics.values():
                belief_metric.add_turn(goal_hyps, label)
            
            
            this_session["turns"].append(goal_hyps)
            
            
        tracker_output["sessions"].append(this_session)
    
    tracker_file = open(args.trackerfile, "wb")
    json.dump(tracker_output, tracker_file, indent=4)
    tracker_file.close()
      
    csv_file = open(args.csv, "wb")
    
    
    output = []
    
    for key, metric in metrics.items():
        this_output =  metric.output()
        for this_key, value in this_output.items():
            output.append(( key + ","+ this_key, value))
            
    for key, belief_metric in belief_metrics.items():
        this_output =  belief_metric.output()
        key = "belief_"+key
        for this_key, value in this_output.items():
            output.append((key + ","+ this_key, value))

    output.sort(key=lambda x:x[0])
    for key, value in output:
        w = 35
        if value < 0 :
            w = w-1
        metric_name = (key+",").ljust(w)    
        csv_file.write(metric_name + ("%.5f"%value)+"\n")
    
    csv_file.close()
        
    

class Fscore(object):
    def __init__(self, ontology):
        self.ontology = ontology
        self.precision_numerator = 0
        self.precision_denominator = 0
        self.recall_numerator = 0
        self.recall_denominator = 0
        
    def add_turn(self, true_label, slu_hyps,  log_turn, label):
        true_tuples = set(filter_informative(uactsToTuples(true_label), self.ontology))
        test_tuples = set(filter_informative(uactsToTuples(slu_hyps[0]["slu-hyp"]), self.ontology))
        
        
        self.precision_numerator += len(true_tuples.intersection(test_tuples))
        self.recall_numerator += len(true_tuples.intersection(test_tuples))
        self.precision_denominator += len(test_tuples)
        self.recall_denominator += len(true_tuples)

    def output(self, ):
        p = float(self.precision_numerator)/self.precision_denominator
        r = float(self.recall_numerator)/self.recall_denominator
        if p*r == 0.0 :
            f = 0.0
        else :
            f = 2*(p*r)/(p+r)
        return {
            "precision":p,
            "recall":r,
            "fscore":f
        }
 
    
class ICE(object):
    def __init__(self, ontology):
        self.ontology = ontology
        self.N = 0.0
        self.Sum = 0.0
        self.regression_data = []
    def add_turn(self, true_label, slu_hyps, log_turn, label):
        
        true_tuples = set(filter_informative(uactsToTuples(true_label), self.ontology))
        test_tuple_scores = defaultdict(float)
        for slu_hyp in slu_hyps:
            test_tuples = set(filter_informative(uactsToTuples(slu_hyp["slu-hyp"]), self.ontology))
            p = slu_hyp["score"]
            for tup in test_tuples:
                test_tuple_scores[tup] += p
        N_delta= len(true_tuples)
        Sum_delta = 0.0
        for tup in true_tuples.union(set(test_tuple_scores)):
            d = int(tup in true_tuples)
            c = 0
            if tup in test_tuple_scores :
                c = test_tuple_scores[tup]
            Sum_delta += math.log(max(eps, d*c + (1-d)*(1-c)))
        
        self.N += N_delta
        self.Sum += Sum_delta
   
    def output(self, ):
        out = {"ICE": -self.Sum/self.N}
        return out
    
    
class BeliefAccuracy(object):
    def __init__(self, ontology):
        self.counters = defaultdict(float)
        self.ontology = ontology
        
    def add_turn(self, goal_hyps, labels):
        
        # goal-labels
        true_labels = {}
        
        for slot in self.ontology["informable"] :
            if  len(self.ontology["informable"][slot])==1:
                continue
            if slot in goal_hyps["goal-labels"]:
                hyps = goal_hyps["goal-labels"][slot]
            else :
                hyps = {}
            hyps = {value:max(0.0,p) for value,p in hyps.items()}
            total_p = sum(hyps.values())
            if total_p > 1.0 :
                hyps = {value: p/total_p for value,p in hyps.items()}
                
            offlist_p = 1.0-total_p
            hyps[None]=offlist_p
            hyps = hyps.items()
            hyps.sort(key = lambda x:-x[1])
            true_goal = None
            if slot in labels["goal-labels"] :
                true_goal = labels["goal-labels"][slot]
            
               
            self.counters["goal_N"] += 1
            # accuracy
            self.counters["goal_acc_corr"] += int(true_goal == hyps[0][0])
            # l2 and logp
            p = 0.0
            qs = []
            for hyp, score in hyps:
                if hyp == true_goal :
                    p = score
                else :
                    qs.append(score)
            self.counters["goal_l2"] += (1-p)**2
            self.counters["goal_logp"] += math.log(max(eps,p))
            self.counters["goal_l2"] += sum([q**2 for q in qs])
            
            
        
        # method-label
        true_method = labels["method-label"]
        method_hyps = goal_hyps["method-label"].items()
        total_p = sum([x[1] for x in method_hyps])
        if total_p < 1.0 :
            method_hyps = [(method, p/total_p) for method,p in method_hyps]
        method_hyps.sort(key = lambda x:-x[1])
        
         # acc
        self.counters["method_N"] += 1
        self.counters["method_acc_corr"] += int(method_hyps[0][0] == true_method)
        
        # l2 and logp
        method_hyps_dict = dict(method_hyps)
        for method in self.ontology["method"]:
            p = 0
            if method in method_hyps_dict :
                p = method_hyps_dict[method]
            if method == true_method :
                self.counters["method_l2"] += (1-p)**2
                self.counters["method_logp"] += math.log(max(eps,p))
            else :
                self.counters["method_l2"] += (p)**2
       
        # requested-slots
        true_requested = labels["requested-slots"]
        for slot in self.ontology["requestable"]:
            self.counters["requested_N"] += 1
            p = 0
            if slot in goal_hyps["requested-slots"]:
                p = goal_hyps["requested-slots"][slot]
            if slot in true_requested :
                self.counters["requested_l2"] += (1-p)**2
                self.counters["requested_logp"] += math.log(max(eps,p))
            else :
                self.counters["requested_l2"] += (p)**2
                self.counters["requested_logp"] += math.log(max(eps,1-p))
            self.counters["requested_acc_corr"] += int((p>0.5)==(slot in true_requested))
            
        
        
    def output(self, ):
        return {
            "goal_acc":self.counters["goal_acc_corr"]/self.counters["goal_N"],
            "goal_l2":self.counters["goal_l2"]/self.counters["goal_N"],
            "goal_logp":self.counters["goal_logp"]/self.counters["goal_N"],
            "method_acc":self.counters["method_acc_corr"]/self.counters["method_N"],
            "method_l2":self.counters["method_l2"]/self.counters["method_N"],
            "method_logp":self.counters["method_logp"]/self.counters["method_N"],
            "requested_acc":self.counters["requested_acc_corr"]/self.counters["requested_N"],
            "requested_l2":self.counters["requested_l2"]/self.counters["requested_N"],
            "requested_logp":self.counters["requested_logp"]/self.counters["requested_N"],
            
            "all_acc":(self.counters["goal_acc_corr"]+self.counters["method_acc_corr"]+self.counters["requested_acc_corr"])/(self.counters["goal_N"]+self.counters["method_N"]+self.counters["requested_N"]),
            "all_l2":(self.counters["goal_l2"]+self.counters["method_l2"]+self.counters["requested_l2"])/(self.counters["goal_N"]+self.counters["method_N"]+self.counters["requested_N"]),
            "all_logp":(self.counters["goal_logp"]+self.counters["method_logp"]+self.counters["requested_logp"])/(self.counters["goal_N"]+self.counters["method_N"]+self.counters["requested_N"]),
            
                }
    
        
    
    

    
def uactsToTuples(uacts):
        out = []
        for uact in uacts:
            act = uact["act"]
            if uact["slots"] == [] :
                out.append((act,))
            for slot,value in uact["slots"]:
                if act == "request" :
                    out.append(("request", value))
                else :
                    out.append((act,slot,value))
        return out
    
    
def filter_informative(tuples, ontology):
    # filter tuples by whether they are informative according to ontology
    new_tuples = []
    for tup in tuples:
        if len(tup) == 3 :
            act, slot, value = tup
            if slot == "this" or (slot in ontology["informable"] and len(ontology["informable"][slot]) > 1) :
                new_tuples.append(tup)
        else :
            new_tuples.append(tup)
    return new_tuples
  




if (__name__ == '__main__'):
    main(sys.argv)

