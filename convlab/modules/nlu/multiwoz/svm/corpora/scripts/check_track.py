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
import os
import sys


def main(argv):
    
    install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    utils_dirname = os.path.join(install_path,'lib')

    sys.path.append(utils_dirname)
    from dataset_walker import dataset_walker
    
    parser = argparse.ArgumentParser(description='Check the validity of a tracker output object.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True,
                        help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True,
                        help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--trackfile',dest='scorefile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing score JSON')
    parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,
                        help='JSON Ontology file')
    
    args = parser.parse_args()

    sessions = dataset_walker(args.dataset,dataroot=args.dataroot,labels=False)
    tracker_output = json.load(open(args.scorefile))
    ontology = json.load(open(args.ontology))
    
    checker = TrackChecker(sessions, tracker_output, ontology)
    checker.check()
    checker.print_errors()
    
    
    
class TrackChecker():
    
    def __init__(self, sessions, tracker_output, ontology):
        self.sessions = sessions
        self.tracker_output = tracker_output
        self.errors = []
        self.ontology = ontology
    
    def check(self):
        # first check the top-level stuff
        if len(self.sessions.datasets) != 1 :
            self.add_error(("top level",), "tracker output should be over a single dataset")
        if "dataset" not in self.tracker_output :
            self.add_error(("top level","trackfile should specify its dataset"))
        elif self.sessions.datasets[0] != self.tracker_output["dataset"]:
            self.add_error(("top level","datasets do not match"))
        if len(self.tracker_output["sessions"]) !=  len(self.sessions) :
            self.add_error(("top level","number of sessions does not match"))
        if "wall-time" not in self.tracker_output :
            self.add_error(("top level","wall-time should be included"))
        else:
            wall_time = self.tracker_output["wall-time"]
            if not isinstance(wall_time, type(0.0)):
                self.add_error(("top level","wall-time must be a float"))
            elif wall_time <= 0.0 :
                self.add_error(("top level","wall-time must be positive"))
                
        # check no extra keys TODO
         
        for session, track_session in zip(self.sessions, self.tracker_output["sessions"]):
            session_id = session.log["session-id"]
            # check session id
            if session_id != track_session["session-id"] :
                self.add_error((session_id,),"session-id does not match")
            # check number of turns
            if len(session) != len(track_session["turns"]) :
                self.add_error((session_id,),"number of turns do not match")
                
            # now iterate through turns
            for turn_num, ((log_turn, label_turn), tracker_turn) in enumerate(zip(session, track_session["turns"])):
                if "method-label" not in tracker_turn :
                    self.add_error((session_id, "turn", turn_num), "no method-label key in turn")
                else :
                    # check method
                    # distribution:
                    self._check_distribution((session_id, "turn", turn_num, "method-label"),
                            tracker_turn["method-label"],
                            self.ontology["method"])
                    
                    
                if "requested-slots" not in tracker_turn :
                    self.add_error((session_id, "turn", turn_num), "no requested-slots key in turn")
                else :
                    # check requested-slots
                    for slot, p in tracker_turn["requested-slots"].items():
                        if slot not in self.ontology["requestable"] :
                            self.add_error((session_id, "turn", turn_num, "requested-slots", slot),
                                "do not recognise requested slot"
                            )
                        if p < 0.0 :
                            self.add_error((session_id, "turn", turn_num, "requested-slots", slot),
                                "score should not be less than 0.0"
                            )
                        elif p > 1.0000001 :
                            self.add_error((session_id, "turn", turn_num, "requested-slots", slot),
                                "score should not be more than 1.0"
                            )
                    
                
                if "goal-labels" not in tracker_turn :
                    self.add_error((session_id, "turn", turn_num), "no goal-labels key in turn")
                else :
                    # check goal-labels
                    for slot, dist in tracker_turn["goal-labels"].items():
                        if slot not in self.ontology["informable"] :
                            self.add_error((session_id, "turn", turn_num, "goal-labels", slot),
                                "do not recognise slot"
                            )
                        else :
                            self._check_distribution((session_id, "turn", turn_num, "goal-labels", slot),
                                    tracker_turn["goal-labels"][slot],
                                    self.ontology["informable"][slot] +['dontcare']
                                )
                        
                    
                
                if "goal-labels-joint" in tracker_turn :
                    # check goal-labels-joint
                    # first check distribution
                    d = {}
                    for i, hyp in enumerate(tracker_turn["goal-labels-joint"]):
                        d[i] = hyp["score"]
                        self._check_distribution(
                            (session_id, "turn", turn_num, "goal-labels-joint", "hyp", i),
                            d
                        )
                    # now check hypotheses
                    for i, hyp in enumerate(tracker_turn["goal-labels-joint"]):
                        for slot in hyp["slots"]:
                            if slot not in self.ontology["informable"] :
                                self.add_error( (session_id, "turn", turn_num, "goal-labels-joint","hyp",i,"slot",slot),
                                    "do not recognise slot"
                                    )
                            else :
                                if hyp["slots"][slot] not in self.ontology["informable"][slot] + ['dontcare'] :
                                    self.add_error( (session_id, "turn", turn_num, "goal-labels-joint","hyp",i,"slot",slot,"value",hyp["slots"][slot]),
                                    "do not recognise slot value"
                                    )
                        
            
    def _check_distribution(self, context, d, valid_values=None) :
        for key, score in d.items():
            if score < 0.0 :
                self.add_error(context+("value",key), "should not be negative")
            elif score > 1.00000001 :
                self.add_error(context+("value",key), "should not be > 1.0")
        total_p = sum(d.values())
        if total_p > 1.000001 :
            self.add_error(context+("total score",), "should not be > 1.0")
        if valid_values != None :
            for value in d.keys():
                if value not in valid_values :
                    self.add_error(context+("value",value), "do not recognise value")
        
        
        
    
    def add_error(self, context, error_str):
        self.errors.append((context, error_str))
    
    
    def print_errors(self):
        if len(self.errors) == 0 :
            print("Found no errors, trackfile is valid")
        else :
            print("Found",len(self.errors),"errors:")
        for context, error in self.errors:
            print(" ".join(map(str, context)), "-", error)
        
    
    
    

    
    
if __name__ =="__main__" :
    main(sys.argv)