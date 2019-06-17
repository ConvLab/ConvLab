# Modified by Microsoft Corporation.
# Licensed under the MIT license.


import configparser
import os
import pprint
import sys
import zipfile

from convlab.modules.nlu.multiwoz.svm import Classifier, sutils


def train(config):
    c = Classifier.classifier(config)
    pprint.pprint(c.tuples.all_tuples)
    print('All tuples:',len(c.tuples.all_tuples))
    model_path = config.get("train", "output")
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('output to {}'.format(model_path))
    dataroot = config.get("train", "dataroot")
    dataset = config.get("train", "dataset")
    dw = sutils.dataset_walker(dataset = dataset, dataroot=dataroot, labels=True)
    c = Classifier.classifier(config)
    c.cacheFeature(dw)
    c.train(dw)
    c.save(model_path)
    with zipfile.ZipFile(os.path.join(model_dir, 'svm_multiwoz.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path)
    
def usage():
    print("usage:")
    print("\t python train.py config/multiwoz.cfg")


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