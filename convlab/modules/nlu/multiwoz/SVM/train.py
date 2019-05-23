# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import configparser, sys
from convlab.modules.nlu.multiwoz.SVM import Classifier, sutils
import pickle
import pprint

def train(config):
    c = Classifier.classifier(config)
    pprint.pprint(c.tuples.all_tuples)
    print(len(c.tuples.all_tuples))
    dataroot = config.get("train", "dataroot")
    dataset = config.get("train", "dataset")
    dw = sutils.dataset_walker(dataset = dataset, dataroot=dataroot, labels=True)
    cache = config.get("train", "cache")
    if cache=='None':
        c = Classifier.classifier(config)
        c.cacheFeature(dw)
        pickle.dump(c,open('cache/cache_sys.pkl','wb'))
    else:
        print("loading cache feature")
        c = pickle.load(open(cache,'rb'))
    # c.cacheFeature(dw)
    c.train(dw)
    c.save(config.get("train", "output"))
    
def usage():
    print("usage:")
    print("\t python train.py config/eg.cfg")


if __name__ == '__main__':
    if len(sys.argv) != 2 :
        usage()
        sys.exit()
        
    config = configparser.ConfigParser()
    try :
        config.read(sys.argv[1])
    except Exception as e:
        print("Failed to parse file")
        print(e)

    train(config)