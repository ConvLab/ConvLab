# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from convlab.modules.nlg.nlg import NLG

class TemplateNLG(NLG):
    def init(self,):
        NLG.__init__(self)

    def generate(self, dialog_act):
        phrases = []
        for da in dialog_act.keys():
            domain, type = da.split('-')
            if domain == 'general':
                if type == 'hello':
                    phrases.append('hello, i need help')
                else:
                    phrases.append('bye')
            elif type == 'Request':
                for slot, value in dialog_act[da]:
                    phrases.append('what is the {}'.format(slot))
            else:
                for slot, value in dialog_act[da]:
                    phrases.append('i want the {} to be {}'.format(slot, value))
        sent = ', '.join(phrases)
        return sent


if __name__ == '__main__':
    nlg = TemplateNLG()
    user_acts = [{"Restaurant-Inform": [["Food", "japanese"], ["Time", "17:45"]]},
                 {"Restaurant-Request": [["Price", "?"]]},
                 {"general-bye": [["none", "none"]]}]
    for ua in user_acts:
        sent = nlg.generate(ua)
        print(sent)
