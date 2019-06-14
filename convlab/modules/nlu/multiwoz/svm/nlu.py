# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import configparser
import os
import zipfile

from convlab.lib.file_util import cached_path
from convlab.modules.nlu.multiwoz.svm import Classifier
from convlab.modules.nlu.nlu import NLU


class SVMNLU(NLU):
    def __init__(self,
                 config_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/multiwoz.cfg'),
                 model_file=None):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.c = Classifier.classifier(self.config)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config.get("train", "output"))
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_path):
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if not model_file:
                print('Load from ', os.path.join(model_dir, 'svm_multiwoz.zip'))
                archive = zipfile.ZipFile(os.path.join(model_dir, 'svm_multiwoz.zip'), 'r')
            else:
                print('Load from model_file param')
                archive_file = cached_path(model_file)
                archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(os.path.dirname(model_dir))
            archive.close()
        self.c.load(model_path)

    def parse(self, utterance, context=None, not_empty=True):
        sentinfo = {
            "turn-id": 0,
            "asr-hyps": [
                    {
                        "asr-hyp": utterance,
                        "score": 0
                    }
                ]
            }
        slu_hyps = self.c.decode_sent(sentinfo, self.config.get("decode", "output"))
        if not_empty:
            act_list = []
            for hyp in slu_hyps:
                if hyp['slu-hyp']:
                    act_list = hyp['slu-hyp']
                    break
        else:
            act_list = slu_hyps[0]['slu-hyp']
        dialog_act = {}
        for act in act_list:
            intent = act['act']
            if intent=='request':
                domain, slot = act['slots'][0][1].split('-')
                intent = domain+'-'+intent.capitalize()
                dialog_act.setdefault(intent,[])
                dialog_act[intent].append([slot,'?'])
            else:
                dialog_act.setdefault(intent, [])
                dialog_act[intent].append(act['slots'][0])
        return dialog_act

if __name__ == "__main__":
    nlu = SVMNLU()
    test_utterances = [
        "What type of accommodations are they. No , i just need their address . Can you tell me if the hotel has internet available ?",
        "What type of accommodations are they.",
        "No , i just need their address .",
        "Can you tell me if the hotel has internet available ?"
        "you're welcome! enjoy your visit! goodbye.",
        "yes. it should be moderately priced.",
        "i want to book a table for 6 at 18:45 on thursday",
        "i will be departing out of stevenage.",
        "What is the Name of attraction ?",
        "Can I get the name of restaurant?",
        "Can I get the address and phone number of the restaurant?",
        "do you have a specific area you want to stay in?"
    ]
    for utt in test_utterances:
        print(utt)
        print(nlu.parse(utt))
