from __future__ import print_function
# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import sys


#############################################################################
#                     Pretty print a specified dialog                       #
#############################################################################

def jsonItemToCued(i):
    if len(i) > 2:
        print("Unsure what to do about " + str(i))
    if len(i) > 1:
        return i[0] + "=" + i[1]
    elif len(i) == 1:
        return i[0]
    else:
        return ""    

def jsonActToCued(a):
    return a["act"] + "(" + ",".join(jsonItemToCued(i) for i in a["slots"]) + ")"

#Converts a json format utterance to a CUED-style string
def jsonToCued(d):
    ans = "|".join(jsonActToCued(a) for a in d)
    return ans

#Pretty prints a dialog
def prettyPrint(fname):
    if os.path.isdir(fname):
        log = json.load(open(os.path.join(fname, "log.json")))
        label = json.load(open(os.path.join(fname, "label.json")))
        for turn, labelturn in zip(log["turns"], label["turns"]) :
            print("SYS  > " + turn['output']['transcript'])
            dact = turn['output']['dialog-acts']
            slulist = turn['input']['live']['slu-hyps']
            print("DAct > " + jsonToCued(dact))
            if len(slulist) > 0:
                for s in slulist:
                    slu = s
                #prob = slulist[0]['prob']
                    print("SLU  > %-20s [%.2f]" % (jsonToCued(slu['slu-hyp']),slu['score']))

            asrlist = turn['input']['live']['asr-hyps']
            print("ASR  > " + asrlist[0]['asr-hyp'])
            print("Tran > " +str(labelturn['transcription']))
            print(" ")
            
            
            

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python prettyPrint.py [dialogfolder]")
    else:
        fname = sys.argv[1]
        prettyPrint(fname)

