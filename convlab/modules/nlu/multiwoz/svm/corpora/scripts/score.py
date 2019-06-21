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
import traceback
from collections import defaultdict

import misc

SCHEDULES = [1,2]
LABEL_SCHEMES = ["a","b"]
EPS = 0.00001

def main(argv):
    
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
    parser.add_argument('--trackfile',dest='scorefile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing score JSON')
    parser.add_argument('--scorefile',dest='csv',action='store',metavar='CSV_FILE',required=True,
                        help='File to write with CSV scoring data')
    parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,
                        help='JSON Ontology file')
    parser.add_argument('--rocdump',dest='rocdump',action='store',metavar='FILE_STEM',
                        help='If present, use this file stem to write out ROC plot data: filestem.<schedule>.<slot>.<type>.csv, where type is either roc (which contains the ROC curve coordinates) or scores (which contains the raw scores used to compute the ROC curves).')

    args = parser.parse_args()

    sessions = dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)
    tracker_output = json.load(open(args.scorefile))
    ontology = json.load(open(args.ontology))
    
    slots_informable  = ontology["informable"].keys()
    slots_requestable = ontology["requestable"]
    
    csvfile = open(args.csv,'w')
    
    # what stats are there?
    stats = []
    stat_classes = [Stat_Accuracy, Stat_Probs, Stat_MRR, Stat_Updates, Stat_ROC]
    
    for schedule in SCHEDULES:
        for label_scheme in LABEL_SCHEMES:
            for component in ['goal','requested', 'method', 'all']:
                if component == 'goal' :
                    for slot in slots_informable + ['all','joint','joint_independent'] :
                        for stat_class in stat_classes:
                            stats.append((('goal', slot), (schedule, label_scheme), stat_class()))
                
                        
                elif component == 'requested' :
                    if label_scheme != "a" :
                        continue
                    for slot in slots_requestable + ['all'] :
                        for stat_class in stat_classes:
                            stats.append((('requested', slot), (schedule, label_scheme), stat_class()))
                            
                elif component == 'method' :
                    for stat_class in stat_classes:
                        stats.append((('method',), (schedule, label_scheme), stat_class()))
                            
                elif component == 'all' :
                    for stat_class in stat_classes:
                        stats.append((('all',), (schedule, label_scheme), stat_class()))
                     
    
    turn_counter = 0.0
    
    for session_num, (session_tracker, session) in enumerate(zip(tracker_output['sessions'], sessions)):
        
      for _, _, stat_class in stats:
          stat_class.newDialog()
            
      session_id = session.log['session-id']
      try:
        
        # these are the set of slots 'mentioned so far', i.e. for schedule2
        S = defaultdict(lambda : set([]))
        S_requested = set([])
        
        session_length = len(session)
        
        goal_labels_b, method_labels_b = misc.LabelsB(session, ontology)
        method_schedule_2 = False # whether schedule 2 is active for method
        
        for turn_num, ((log_turn,label_turn),_tracker_turn) in enumerate(zip(session,session_tracker['turns'])):
            turn_counter += 1.0
            S_new = misc.S(log_turn, ontology)
            
            for slot in S_new :
                S[slot] = S[slot].union(S_new[slot])
                
            # remove just informed slots from S_requested
            S_requested = S_requested.difference(misc.SysInformed(log_turn))
            # add in ones from slu hyps
            S_requested = S_requested.union(set(misc.S_requested(log_turn)))

            tracker_goal_labels = _tracker_turn["goal-labels"]
            for slot in slots_informable:
                if slot in tracker_goal_labels :
                    tracker_goal_labels[slot] = normalise_dist(tracker_goal_labels[slot].items(), (session_id, turn_num, "goal."+slot))
                else :
                    tracker_goal_labels[slot] = [(None, 1.0)]
            
            
            # prepare for joint goals scoring:
            tracker_goal_joint_labels = "independent"
            if "goal-labels-joint" in _tracker_turn :
                tracker_goal_joint_labels = _tracker_turn["goal-labels-joint"]
                
            if tracker_goal_joint_labels != "independent" :
                # tracker_goal_joint_labels must be a list of joint hyps
                tracker_goal_joint_labels = [(hyp["slots"], hyp["score"]) for hyp in tracker_goal_joint_labels]
                tracker_goal_joint_labels = normalise_dist(tracker_goal_joint_labels, (session_id, turn_num, "goal.joint"))
            
            # also gather the correct joint label
            true_goal_joint = None
            for slot in label_turn["goal-labels"]:
                if true_goal_joint == None :
                    true_goal_joint = {}
                true_goal_joint[slot] = label_turn["goal-labels"][slot]
            
            true_goal_joint_b = None
            for slot in goal_labels_b[turn_num]:
                if true_goal_joint_b == None :
                    true_goal_joint_b = {}
                true_goal_joint_b[slot] = goal_labels_b[turn_num][slot]
            
            
            tracker_requested_slots = _tracker_turn["requested-slots"]
            for slot in tracker_requested_slots:
                dist = [(True, tracker_requested_slots[slot]), (False,1.0-tracker_requested_slots[slot])]
                tracker_requested_slots[slot] = normalise_dist(dist, (session_id, turn_num, "requested."+slot))
            
            tracker_method_label = normalise_dist(_tracker_turn["method-label"].items(), (session_id, turn_num,"method"))
            
            # for method schedule 2, work out whether any slu-hyp has been given
            # which informs the method:
            
            if not method_schedule_2 :
                mact = log_turn["output"]["dialog-acts"]
                for slu_hyp in log_turn["input"]["live"]["slu-hyps"] :
                    user_act = slu_hyp["slu-hyp"]
                    method_label = misc.MethodLabel(user_act, mact)
                    if method_label != "none" :
                        method_schedule_2 = True
                        break
                    
            
            for component, (schedule, label_scheme), stat_class in stats:
                if component[0] == "goal" and (component[1] == "joint" or  component[1] == "joint_independent"):
                    if schedule == 2:
                        # calculate schedule2 applicability
                        applies = False
                        for slot in slots_informable:
                            if len(S[slot]) > 0:
                                applies = True
                                break
                        if not applies :
                            continue
                        
                    this_true_label = true_goal_joint
                    if label_scheme == "b" :
                        this_true_label = true_goal_joint_b
                    
                    if tracker_goal_joint_labels == "independent" or component[1] == "joint_independent" :
                        stat_class.add(tracker_goal_labels, this_true_label, (session_id, turn_num, component, schedule, label_scheme), independent=True)
                    else :
                        stat_class.add(tracker_goal_joint_labels, this_true_label, (session_id, turn_num, component, schedule, label_scheme))
                
                if (component[0] == "goal" or component[0] == "all") and (len(component)==1 or ("joint" not in component[1])) :
                    if component[0] == "all" or component[1] == "all" :
                        slots = slots_informable[:]
                    else :
                        slots = [component[1]]
                    for slot in slots:
                        if schedule ==2 and len(S[slot]) == 0 :
                            continue
                        dist = tracker_goal_labels[slot]
                        
                        true_label = None
                        if slot in label_turn["goal-labels"] :
                            true_label = label_turn["goal-labels"][slot]
                            
                        if label_scheme == "b" :
                            true_label = None
                            if slot in goal_labels_b[turn_num] :
                                true_label = goal_labels_b[turn_num][slot]
                            
                        stat_class.add(dist, true_label, (session_id, turn_num, component, schedule, label_scheme))
                
                
                if component[0] == "requested" or component[0] == "all" :
                    if  component[0] == "all" or  component[1] == "all":
                        slots = slots_requestable[:]
                    else :
                        slots = [component[1]]
                    for slot in slots:
                        if schedule ==2 and (slot not in S_requested):
                            continue
                        dist =  [(False,1.0), (True,0.0)]
                        if  slot in tracker_requested_slots :
                            dist = tracker_requested_slots[slot]
                        
                        true_label = (slot in label_turn["requested-slots"])                        
                        stat_class.add(dist, true_label, (session_id, turn_num, component, schedule, label_scheme))
                        
                        
                if component[0] == "method" or component[0] == "all":
                    if schedule == 2 and not method_schedule_2:
                        continue # no slu hyp informing the method has been given yet.
                    dist = tracker_method_label
                    true_label =  label_turn["method-label"]
                    if label_scheme == "b" :
                        true_label = method_labels_b[turn_num]
                        
                        
                    stat_class.add(dist, true_label, (session_id, turn_num, component, schedule, label_scheme))
      except KeyboardInterrupt :
          raise
      except:
          traceback.print_exc(file=sys.stdout)
          print("While scoring " + str(session_id))
    # output to csv
    print(( "state_component, stat, schedule, label_scheme, N, result"), file=csvfile)
    
    for stat in stats:
        component, (schedule, label_scheme), stat_class = stat
        results = stat_class.results()
        for stat_subname, N, result in results:
            if result == None :
                result = "-"
            else :
                result = "%.7f"%result
            print(( "%s, %s, %i, %s, %i, %s"%(".".join(component), stat_subname, schedule, label_scheme, N, result)), file=csvfile)
        if isinstance(stat_class, Stat_ROC) and (args.rocdump):
            rocfile = args.rocdump + '.schedule' + str(schedule) + str(label_scheme)+'.' + (".".join(component)) + '.roc.csv'
            scoresfile = args.rocdump + '.schedule' + str(schedule) + str(label_scheme)+'.' + (".".join(component)) + '.scores.csv'
            stat_class.DumpROCToFile(rocfile)
            stat_class.DumpScoresToFile(scoresfile)
        
    print('basic,total_wall_time,,,,%s' % (tracker_output['wall-time']), file=csvfile)
    print('basic,sessions,,,,%s' % (len(sessions)), file=csvfile)
    print('basic,turns,,,,%i' % (int(turn_counter)), file=csvfile)
    print('basic,wall_time_per_turn,,,,%s' % (tracker_output['wall-time'] / turn_counter), file=csvfile)
    print('basic,dataset,,,,%s' % (tracker_output['dataset'] ), file=csvfile)

    csvfile.close()
    


def normalise_dist(dist, this_id=None):
    # take dist , convert to a new list of tuples, ordered and made to sum up to
    # no more than 1
    out = dist[:]
    
    context_string = ""
    if this_id != None :
        context_string = this_id[0] + (", turn %i" % this_id[1]) + ", "+this_id[2] 
        
    for i in range(len(out)):
        if out[i][1] < 0.0 :
            print('WARNING: Score is less than 0.0, changing to 0.0',context_string, file=sys.stderr)
    
    total_p = sum([x[1] for x in out])
    if total_p >1.0 :
        if abs(total_p - 1.0) > EPS :
            print('WARNING: scores sum to more than 1, renormalising',context_string, file=sys.stderr)
        out = [(x[0],x[1]/total_p) for x in out]
        total_p = 1.0
        
    out.append((None, 1.0-total_p))
    
    out.sort(key = lambda x:-x[1])
    return out

class Stat(object):
    def __init__(self, ):
        pass
        
    def add(self, dist,  true_label, this_id, independent=False):
        pass
    
    def results(self, ):
        return []
    
    def newDialog(self) :
        return
    
 
class Stat_Accuracy(Stat):
    def __init__(self, ):
        self.N = 0.0
        self.correct = 0.0
        
    def add(self, dist,  true_label, this_id, independent=False):
        if independent :
            top_hyp, _ = tophyp_independent(dist)
            self.correct += int(top_hyp == true_label)
        else :
            self.correct += int(dist[0][0]== true_label)
        self.N += 1
    
    def results(self, ):
        acc = None
        if self.N > 0.0:
            acc = self.correct/self.N
        return [
            ("acc", self.N, acc)
        ]
    
    
    
class Stat_MRR(Stat):
    def __init__(self, ):
        self.N = 0.0
        self.numerator = 0.0
        
    def add(self, dist,  true_label, this_id, independent=False):
        recip_rank = 0.0
        if independent :
            ranks = []
            for slot in dist:
                found = False
                for i, (hyp, _) in enumerate(dist[slot]):
                    if ((true_label == None or slot not in true_label) and hyp == None) or (true_label != None and slot in true_label and hyp == true_label[slot]) :
                        ranks.append(i)
                        found = True
                        break
                if not found :
                    ranks.append(None)
            
            if None in ranks :
                recip_rank = 0.0
            else :
                rank = 1.0
                for r in ranks:
                    rank *= (1+r)
                recip_rank = 1.0/rank
                
            
        else :
            
            for i, (hyp, _) in enumerate(dist):
                if hyp == true_label :
                    recip_rank = 1.0/(1.0+i)
                    break
        self.numerator += recip_rank
        self.N += 1
    
    def results(self, ):
        mrr = None
        if self.N > 0.0:
            mrr = self.numerator/self.N
        return [
            ("mrr", self.N, mrr)
        ]
    
class Stat_Probs(Stat):
    def __init__(self, ):
        self.N = 0.0
        self.numerator_l2 = 0.0
        self.numerator_brier = 0.0
        self.numerator_avgp = 0.0
        self.numerator_neglogp = 0.0
        self.dialog_acc = []
        
    def add(self, dist,  true_label, this_id,independent=False):
        if independent :
            ps = []
            for slot in dist:
                found = False
                for (hyp, score) in dist[slot]:
                    if ((true_label == None or slot not in true_label) and hyp == None) or (true_label != None and slot in true_label and hyp == true_label[slot]) :
                        ps.append(score)
                        found = True
                if not found :
                    ps.append(0.0)
                    
            p = 1.0
            for p_ in ps:
                p *= p_
                
            sum_q = 1-p
            sum_q2 = 1.0
            for slot in dist:
                sum_q2 *= sum([score**2 for hyp_,score in dist[slot]])
            sum_q2 = sum_q2 - p**2
            
            self.numerator_l2 += (1-p)**2 + sum_q2
            self.numerator_brier += (1-p)**2 +(sum_q)**2
            self.numerator_avgp += p
            self.numerator_neglogp += -math.log(max(0.0001, p))
            
            
        else :
            p = 0.0
            
            qs = []
            for hyp, _p in dist:
                if hyp == true_label :
                    p = _p
                else :
                    qs.append(_p)
            
            self.numerator_l2 += (1-p)**2 + sum([q**2 for q in qs])
            self.numerator_brier += (1-p)**2 + sum(qs)**2
            self.numerator_avgp += p
            self.numerator_neglogp += -math.log(max(0.0001, p))
        self.N += 1
    
    def results(self, ):
        l2 = None
        brier = None
        avgp = None
        neglogp = None
        if self.N > 0.0:
            l2 = self.numerator_l2/self.N
            brier = self.numerator_brier/self.N
            avgp = self.numerator_avgp/self.N
            neglogp = self.numerator_neglogp/self.N
            
        return [
            ("l2", self.N, l2),
            ("l2.binary", self.N, brier),
            ("avgp", self.N, avgp),
            ("neglogp", self.N, neglogp),
        ]
    
    
class Stat_Updates(Stat):
    def __init__(self, ):
        # page 10 of R Higashinaka et al,
        # Evaluating Discourse Understanding in Spoken Dialogue Systems
        self.N = 0.0
        self.correct_updates = 0.0
        self.update_insertions = 0.0
        self.update_substitutions = 0.0
        self.update_deletions = 0.0
        
    def add(self, dist,  true_label, this_id, independent=False):
        
        
        
        if independent :
            current, _ = tophyp_independent(dist)
        else :
            current = dist[0][0]
            
        self.correct_updates += int((self.previous != true_label) \
                                and (self.previous != current) \
                                and (true_label == current))
        
        self.update_insertions += int((self.previous == true_label) \
                                and (self.previous != current) )
        
        self.update_substitutions += int((self.previous != true_label) \
                                and (self.previous != current) \
                                and (true_label != current))
        
        self.update_deletions += int((self.previous != true_label) \
                                and (self.previous == current) )
        
        self.previous = current
            
        self.N += 1
    
    def results(self, ):
        acc = None
        prec = None
        acc_denom = (self.correct_updates+self.update_substitutions+self.update_deletions)
        prec_denom = (self.correct_updates+self.update_substitutions+self.update_insertions)
        if acc_denom > 0 :
            acc = self.correct_updates/acc_denom
        if prec_denom > 0:
            prec = self.correct_updates/prec_denom
        return [
            ("update.acc", self.N, acc),
            ("update.prec", self.N, prec),
        ]
    
    def newDialog(self) :
        self.previous = None

def _changingIndices(x) :
    out = [0]
    value = x[0]
    for i, x_value in enumerate(x) :
        if x_value != value :
            out.append(i)
            value = x_value
    return out

def _cumSum(x) :
    out = []
    cum = 0.0
    for x_value in x:
        cum += x_value
        out.append(cum)
    return out

class Stat_ROC(Stat):
    def __init__(self):
        self.data = []
        self.N = 0
        
    def add(self, dist,  true_label, this_id, independent=False):
        
        if independent :
            top_hyp, score = tophyp_independent(dist)
            label = top_hyp == true_label
            
        else :
            label = dist[0][0]== true_label
            score = dist[0][1]
            
        self.data.append(
            (label, score)
        )
        self.N = len(self.data)
    
    def results(self, ):
        self._calculateROC()
        
        return (
             ('roc.v1_eer', self.N, self.EER() ),
             ('roc.v1_ca05', self.N, self.CA_at_FA(0.05) ),
             ('roc.v1_ca10', self.N, self.CA_at_FA(0.10) ),
             ('roc.v1_ca20', self.N, self.CA_at_FA(0.20) ),
             ('roc.v2_ca05', self.N, self.CA_at_FA(0.05,version=2) ),
             ('roc.v2_ca10', self.N, self.CA_at_FA(0.10,version=2) ),
             ('roc.v2_ca20', self.N, self.CA_at_FA(0.20,version=2) ),
            )
    def EER(self) :
        if (self.N < 2):
            return None
        for (t,ta,fa,tr,fr) in self.roc_curve:
            if (fr >= fa):
                return float(fr + fa)/self.N
        print('Could not find a place where FR >= FA')
        return None
    
    def _calculateROC(self) :
        self.data.sort(key=lambda x:-x[1])
        N = len(self.data)
        if N <= 2 :
            self.roc_curve = []
            return
        indices = _changingIndices([x[1] for x in self.data[:-1]]) + [N-1]
        # true/false accepts/rejects
        cumsum = _cumSum([int(x[0]) for x in self.data])
        N_true = sum([int(x[0]) for x in self.data])
        N_false = N-N_true
        frs = [N_true-cumsum[i] for i in indices]
        trs = [N_false-i+cumsum[i]-1 for i in indices]
        fas = [i-cumsum[i]+1 for i in indices]
        tas =  [cumsum[i] for i in indices]
        thresholds = [self.data[i][1] for i in indices]
        self.roc_curve = zip(thresholds,tas, fas, trs, frs)
        self.roc_curve.reverse() # so thresholds are increasing
        
    def CA_at_FA(self,fa_thresh,version=1):
        assert (version in [1,2]),'Dont know version %s' % (version)
        if (self.N < 2):
            return None
        if (version == 1):
            for (t,ta,fa,tr,fr) in self.roc_curve:
                if (float(fa)/self.N <= fa_thresh):
                    return float(ta)/self.N
            print('Could not find a place where FA <= FA_THRESH')
            return None
        else:
            for (t,ta,fa,tr,fr) in self.roc_curve:
                try :
                    ta_rate = ta/(ta + fr)
                    fa_rate = fa/(fa + tr)
                    if (fa_rate <= fa_thresh):
                        return ta_rate
                except ZeroDivisionError :
                    continue
            return None
        




    def DumpROCToFile(self,filename):
        pass

    def DumpScoresToFile(self,filename):
        print("creating", filename)
        f = open(filename,'w')
        print('label,score', file=f)
        for label, score in self.data:
            print('%s,%s'%(label,score), file=f)
        f.close()
    
def tophyp_independent(dists) :
    top_hyp = None
    top_score = 1.0
    for slot in dists :
        top,score = dists[slot][0]
        if top != None:
            if top_hyp == None :
                top_hyp = {}
            top_hyp[slot] = top
        top_score *= score
    return (top_hyp, top_score)
        

            

if (__name__ == '__main__'):
    main(sys.argv)

