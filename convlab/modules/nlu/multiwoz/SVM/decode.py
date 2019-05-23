# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import configparser, sys
from convlab.modules.nlu.multiwoz.SVM import Classifier, sutils


def decodeToFile(config):
    c = Classifier.classifier(config)
    c.load(config.get("train", "output"))
    
    dataroot = config.get("decode", "dataroot")
    dataset = config.get("decode", "dataset")
    dw = sutils.dataset_walker(dataset = dataset, dataroot=dataroot, labels=False)
    
    c.decodeToFile(dw, config.get("decode","output"))
    
def usage():
    print("usage:")
    print("\t python decode.py config/eg.cfg")

def init_classifier(config):
    c = Classifier.classifier(config)
    c.load(config.get("train", "output"))
    return c

def decode(c,config,sentinfo):
    slu_hyps=c.decode_sent(sentinfo, config.get("decode","output"))
    return slu_hyps

def testing_currTurn():
            sentinfo={
            "turn-id": 2,
            "asr-hyps": [
                    {
                        "asr-hyp": "The Cambridge Belfry is located in the west and rated 4 stars .",
                        "score": 0
                    }
                ]
            }
            return sentinfo

if __name__ == '__main__':

    if len(sys.argv) != 2 :
        print(len(sys.argv))
        print(sys.argv)
        usage()
        sys.exit()

    config = configparser.ConfigParser()
    try :
         config.read(sys.argv[1])
    except Exception as e:
        print("Failed to parse file")
        print(e)

    # decodeToFile(config)

    sentinfo=testing_currTurn()
    slu_hyps=decode(init_classifier(config),config, sentinfo)
    for hyp in slu_hyps:
        print(hyp)