
from convlab.modules.util.multiwoz.dbquery import query
from convlab.modules.util.multiwoz.nlp import normalize, delexicalise
import random
import re
import os
import pickle
import json

domains = ("restaurant", "hotel", "train", "attraction", "taxi", "hospital", "police")

class Preprocessor(object):
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), 'svdic.pkl'), 'rb') as f:
            self.dic = pickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), "belief_state.json")) as f:
            self.slots = json.load(f)

    def process(self, sent: str, state_info: dict, last_domain: str):
        sent = normalize(sent)
        words =  sent.split()

        # replace value appeared in database to placeholder
        sent = delexicalise(' '.join(words), self.dic)
        domain = self._judge_domain(sent, state_info, last_domain)

        # fix placeholder
        if domain!="unk":
            sent = self._fixDelex(sent, domain)

        sent = re.sub(r'\d+', "[value_count]", sent)

        return sent, domain

    def _judge_domain(self, sent, state_info, domain):
        if domain != "unk" and domain in domains:
            return domain

        # judge domain by sentence
        if domain == "unk":
            for word in sent.split(' '):
                if ']' not in word or 'food' in word:
                    for dom in domains:
                        if dom in word:
                            domain = dom
                            break
                    if domain!="unk":
                        break
                    if word=="car":
                        domain = "taxi"
                        break

        return domain

    def _fixDelex(self, sent, domain):
        if domain=="attraction":
            sent = sent.replace("restaurant_", "attraction_").replace("hotel_", "attraction_")
        elif domain=="hotel":
            sent = sent.replace("attraction_", "hotel_").replace("restaurant_", "hotel_")
        elif domain=="restaurant":
            sent = sent.replace("attraction_", "restaurant_").replace("hotel_", "restaurant_")
        return sent

class Dialog_manager(object):
    def __init__(self):
        self.total_states = {dom:{} for dom in domains}   # the belief state for the dialogue
        self.cur_result = {dom:None for dom in domains}   # the query result from the database
        self.cur_result_lower = {dom:None for dom in domains}  # query result, lower case version
        self.number = 0      # the number of all results that satisfies the user goal
        
        self.preprocessor = Preprocessor()
        self.domain = domains[7] # unk

    def reset(self):
        self.total_states = {dom:{} for dom in domains}
        self.cur_result = {dom:None for dom in domains}
        self.cur_result_lower = {dom:None for dom in domains}
        self.number = 0
        self.domain = domains[7] # unk

    def parse(self, states):
        parsed_states = {}
        domain = 'unk'
        for dom, info in states.items():
            for slot, value in info['semi'].items():
                if value !='' and dom+'-'+slot in self.preprocessor.slots:
                    parsed_states[dom+'-'+slot]=value
                    domain = dom
            for slot, value in info['book'].items():
                if value !='' and slot in ['people', 'stay']:
                    parsed_states[dom+'-'+slot]=value
                    domain = dom
        return parsed_states, domain

    def process(self, sentence, state_info, domain):
        '''preprocess the data to match the model input
        '''

        # replace value to placeholder by state tracker
        if domain != "unk":
            for k, v in state_info.items():
                dom, slot = k.lower().split('-')
                if slot=="people" or slot=="stay":
                    continue
                sentence = sentence.replace(v+' ', '[{}_{}] '.format(dom, slot))

        # preprocess
        sentence, domain = self.preprocessor.process(sentence, state_info, domain)

        # judge domain by last domain 
        if domain == "unk":
            if "?" in sentence or "please" in sentence or "need" in sentence:
                domain = self.domain
            elif "thank" not in sentence and " all " not in sentence:
                domain = self.domain
        self.domain = domain

        return sentence, self.domain

    def update(self, states, domain):
        # modify the values in states
        new_states = dict()
        for k, v in states.items():
            slot = k.split('-')[1]
            if slot == 'type' and v.endswith('s'):
                new_states[k] = v.rstrip('s')
            else:
                new_states[k] = v
        states = new_states

        # update belief states
        if domain=="unk":
            return self.total_states, 0, None
        self.total_states[domain].update(states)

        # generate query constraints
        constraints = []
        for slot, value in self.total_states[domain].items():
            pairs = (slot.split('-')[1], value.lower())
            if slot.lower() != 'train-day' and pairs[0].lower() in ['people', 'stay', 'day']:
                continue
            constraints.append(pairs)

        # taxi domain can't not query twice
        if domain=="taxi" and self.cur_result[domain] is not None:
            return self.number, self.cur_result[domain]

        # query results
        results = query(domain, constraints)
        self.number = len(results)

        # update result if old result is not satisfied
        if self.number > 0:
            if self.cur_result[domain] is None or self.cur_result[domain] not in results:
                self.cur_result[domain] = results[0]
        else:
            self.cur_result[domain] = None
        if self.cur_result[domain] is not None:
            result_lower = {}
            for k,v in self.cur_result[domain].items():
                result_lower[k.lower()] = v
            self.cur_result_lower[domain] = result_lower
        else:
            self.cur_result_lower[domain] = None

        return self.number, self.cur_result[domain]

    def relexilize(self, sentence, domain):
        '''used to fullfill the placeholder in the generated sentence
        according to the query results or the belief state
        '''
        sentence = sentence.replace('[UNK]', 'unknown')
        words = sentence.split(' ')
        res = []
        replace_no = False
        for i in range(len(words)):
            word = words[i]
            if word[0] == '[' and word[-1] == ']':
                dom, slot = word[1:-1].split('_')
                if slot=="type":
                    slot = "types"
                elif slot=="reference":
                    slot = "ref"

                # fullfill the domain specific value
                if dom in domains:
                    if self.cur_result_lower[dom] is not None:
                        if slot in self.cur_result_lower[dom]:
                            if slot=="price":
                                res.append(self.cur_result_lower[dom][slot].split(' ')[0])
                            else:
                                res.append(self.cur_result_lower[dom][slot])
                        elif dom+'_'+slot in self.cur_result_lower[dom]:
                            value = self.cur_result_lower[dom][dom+'_'+slot]
                            if isinstance(value, list):
                                value = ''.join(str(t) for t in value)
                            res.append(value)
                        elif self.total_states is not None and dom+'-'+slot in self.total_states[dom]:
                            res.append(self.total_states[dom][dom+'-'+slot])
                        else:
                            res.append(word)
                    elif self.total_states is not None and dom+'-'+slot in self.total_states[dom]:
                        res.append(self.total_states[dom][dom+'-'+slot])
                    else:
                        res.append(word)

                # fullfil the generate value such as value count
                else:
                    if dom == "value" and slot == "count":
                        if words[i+1] in ("is", "of") or words[i-1]=="book":
                            res.append("one")
                        elif words[i+1] in ("minute", "minutes") and domain=="train":
                            if self.cur_result_lower["train"] is not None:
                                res.append(self.cur_result_lower["train"]["duration"].split(' ')[0])
                            else:
                                res.append("0")
                        elif words[i+1] in ("star", "stars") and domain=="hotel":
                            if self.cur_result_lower["hotel"] is not None:
                                res.append(self.cur_result_lower["hotel"]["stars"])
                            else:
                                res.append("0")
                        elif words[i+1] in ("tickets", "seats") and domain=="train" and "train-people" in self.total_states[domain]:
                            res.append(str(self.total_states[domain]["train-people"]))
                        elif words[i+1] in ("nights") and domain=="hotel" and "hotel-day" in self.total_states[domain]:
                            res.append(str(self.total_states[domain]["hotel-day"]))
                        elif self.number==0:
                            res.append("no")
                            replace_no = True
                        else:
                            res.append(str(self.number))
                    elif domain!="unk" and self.cur_result_lower[domain] is not None and slot in self.cur_result_lower[domain]:
                        res.append(self.cur_result_lower[domain][slot])
                    else:
                        res.append(word)
            else:
                res.append(word)

        sentence = ' '.join(res)
        
        # if there is no results for current user goal, rewrite the generated sentence
        if replace_no and "." in res and domain!="unk":
            sentence = sentence.split('.')[0]+'.'
            # randomly select a user provided slot to let user changes his goal
            slots = []
            for slot in [keys.split('-')[1] for keys in self.total_states[domain]]:
                if slot=="food":
                    slots.append("type of food")
                elif slot=="pricerange":
                    slots.append("price range")
                else:
                    slots.append(slot)
            if len(slots)==0:
                slots=["day"]
            sentence = "i am sorry , "+sentence+" would you like to try a different "+slots[random.randint(0, len(slots)-1)]
        sentence = re.sub(r'\[[a-z]+_[a-z]+\]', 'unknown', sentence)
        
        def denormalize(uttr):
            uttr = uttr.replace(' -s', 's')
            uttr = uttr.replace('expensive -ly', 'expensive')
            uttr = uttr.replace('cheap -ly', 'cheap')
            uttr = uttr.replace('moderate -ly', 'moderately')
            uttr = uttr.replace(' -er', 'er')
            uttr = uttr.replace('_UNK', 'unknown')
            return uttr
        sentence = denormalize(sentence)

        return sentence






