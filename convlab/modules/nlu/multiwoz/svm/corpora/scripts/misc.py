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

from collections import defaultdict

import baseline

slots_informable = ["area","food","pricerange","name", "hastv", "hasinternet", "childrenallowed", "near", "hasmusic", "type"]


def S(turn, ontology=None) :
    if ontology == None :
        _slots_informable = slots_informable[:]
    else :
        _slots_informable = ontology["informable"].keys()
        
    mact = []
    if "dialog-acts" in turn["output"] :
        mact = turn["output"]["dialog-acts"]
    this_slot = None
    for act in mact :
        if act["act"] in ["request"]:
            this_slot = act["slots"][0][1]
        elif act["act"] in ["expl-conf", "select"]:
            this_slot = act["slots"][0][0]
    # return a dict of informable slots to mentioned values in a turn
    out = defaultdict(set)
    for act in mact :
        if "conf" in act["act"]  :
            for slot, value in act["slots"] :
                if slot in _slots_informable :
                    out[slot].add(value)

    for slu_hyp in turn["input"]["live"]["slu-hyps"] :
        for act in slu_hyp["slu-hyp"] :
            for slot, value in act["slots"] :
                if slot == "this" :
                    slot = this_slot
                if slot in _slots_informable :
                    out[slot].add(value)

    return out


def S_requested(turn):
    # which slots are hypothesised to be requested in this turn?
    requested = []
    for slu_hyp in turn["input"]["live"]["slu-hyps"]:
        for act in slu_hyp["slu-hyp"] :
            if act["act"] != "request" :
                continue
            for slot, value in act["slots"]:
                if slot == "slot" :
                    requested.append(value)

    return requested


def SysInformed(turn):
    # which slots are informed in this turn?
    informed = set([])
    macts = []
    if "dialog-acts" in turn["output"] :
        macts = turn["output"]["dialog-acts"]
    for mact in macts:
        if mact['act'] == 'inform' or mact['act']== 'offer' :
            for slot, _value in mact['slots']:
                informed.add(slot)
    return informed

def MethodLabel(user_act, mact) :
    method="none"
    act_types = [act["act"] for act in user_act]
    mact_types = [act["act"] for act in mact]
    if "reqalts" in act_types :
        method = "byalternatives"
    elif "bye" in act_types :
        method = "finished"
    elif "inform" in act_types:
        method = "byconstraints"
        for act in [uact for uact in user_act if uact["act"] == "inform"] :
            slots = [slot for slot, _ in act["slots"]]
            if "name" in slots :
                method = "byname"
    return method

def LabelsB(session, ontology) :
    # calculate labelling scheme B labels: (for goal and method)
    goal_labels_b = []
    method_labels_b = []
    slots_informable = ontology["informable"].keys()
    canthelped = {slot:[] for slot in slots_informable}
    for (log_turn,label_turn) in session :
        user_act = label_turn["semantics"]["json"]
        mact = log_turn["output"]["dialog-acts"]
        _, _, _, new_method = baseline.labels(user_act, mact)
        method_label = new_method
        method_labels_b.append(method_label)
        
        if method_label != None :
            for i in range(len(method_labels_b)-1) :
                if method_labels_b[i] == None :
                    method_labels_b[i] = method_label
        
        this_label = {}
        
        for dact in log_turn["output"]["dialog-acts"] :
            if dact["act"] == "canthelp" :
                for slot, value in dact["slots"] :
                    canthelped[slot].append(value)
                    
        for slot in slots_informable :
            
                        
            if slot in label_turn["goal-labels"] and label_turn["goal-labels"][slot] not in canthelped[slot] :
                this_label[slot] = label_turn["goal-labels"][slot]
                for _label in goal_labels_b :
                    if _label[slot] == None :
                        _label[slot] = label_turn["goal-labels"][slot]
            else :
                this_label[slot] = None
                        
        goal_labels_b.append(this_label)
        
    for i in range(len(goal_labels_b)):
        to_delete = []
        for slot in goal_labels_b[i] :
            if goal_labels_b[i][slot] == None:
                to_delete.append(slot)
        for slot in to_delete :
            del goal_labels_b[i][slot]
            
    return goal_labels_b, method_labels_b


if __name__ == '__main__':
    pass