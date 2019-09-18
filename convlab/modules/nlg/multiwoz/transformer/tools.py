from convlab.modules.nlg.multiwoz.transformer.transformer import Constants
import json
import math
from collections import Counter
from nltk.util import ngrams
import numpy
import torch
import os

def get_n_params(*params_list):
    pp=0
    for params in params_list:
        for p in params:
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
    return pp

def filter_sents(sents, END):
    hyps = []
    for batch_id in range(len(sents)):
        done = False
        for beam_id in range(len(sents[batch_id])):
            sent = sents[batch_id][beam_id]
            for s in sent[::-1]:
                if s in [Constants.PAD, Constants.EOS]:
                    pass
                elif s in END:
                    done = True
                    break
                elif s not in END:
                    done = False
                    break
            if done:
                hyps.append(sent)
                break
        if len(hyps) < batch_id + 1:
            hyps.append(sents[batch_id][0])
    return hyps

def obtain_TP_TN_FN_FP(pred, act, TP, TN, FN, FP, elem_wise=False):
    if isinstance(pred, torch.Tensor):
        if elem_wise:
            TP += ((pred.data == 1) & (act.data == 1)).sum(0)
            TN += ((pred.data == 0) & (act.data == 0)).sum(0)
            FN += ((pred.data == 0) & (act.data == 1)).sum(0)
            FP += ((pred.data == 1) & (act.data == 0)).sum(0)
        else:
            TP += ((pred.data == 1) & (act.data == 1)).cpu().sum().item()
            TN += ((pred.data == 0) & (act.data == 0)).cpu().sum().item()
            FN += ((pred.data == 0) & (act.data == 1)).cpu().sum().item()
            FP += ((pred.data == 1) & (act.data == 0)).cpu().sum().item()
        return TP, TN, FN, FP
    else:
        TP += ((pred > 0).astype('long') & (act > 0).astype('long')).sum()
        TN += ((pred == 0).astype('long') & (act == 0).astype('long')).sum()
        FN += ((pred == 0).astype('long') & (act > 0).astype('long')).sum()
        FP += ((pred > 0).astype('long') & (act == 0).astype('long')).sum()
        return TP, TN, FN, FP
    
class F1Scorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, hypothesis, corpus, n=1):
        # containers
        data_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(data_dir, 'data/placeholder.json')) as f:
            placeholder = json.load(f)['placeholder']
        
        TP, TN, FN, FP = 0, 0, 0, 0
        # accumulate ngram statistics
        files = hypothesis.keys()
        for f in files:
            hyps = hypothesis[f]
            refs = corpus[f]        

            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            # Shawn's evaluation
            #refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            #hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']
            for hyp, ref in zip(hyps, refs):
                pred = numpy.zeros((len(placeholder), ), 'float32')
                gt = numpy.zeros((len(placeholder), ), 'float32')
                for h in hyp:
                    if h in placeholder:
                        pred[placeholder.index(h)] += 1
                for r in ref:
                    if r in placeholder:
                        gt[placeholder.index(r)] += 1
                TP, TN, FN, FP = obtain_TP_TN_FN_FP(pred, gt, TP, TN, FN, FP)
            
        precision = TP / (TP + FP + 0.001)
        recall = TP / (TP + FN + 0.001)
        F1 = 2 * precision * recall / (precision + recall + 0.001)            
        return F1

def sentenceBLEU(hyps, refs, n=1):   
    count = [0, 0, 0, 0]
    clip_count = [0, 0, 0, 0]
    r = 0
    c = 0
    weights = [0.25, 0.25, 0.25, 0.25]
    hyps = [hyp.split() for hyp in hyps]
    refs = [ref.split() for ref in refs]
    # Shawn's evaluation
    refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
    hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']
    for idx, hyp in enumerate(hyps):
        for i in range(4):
            # accumulate ngram counts
            hypcnts = Counter(ngrams(hyp, i + 1))
            cnt = sum(hypcnts.values())
            count[i] += cnt

            # compute clipped counts
            max_counts = {}
            for ref in refs:
                refcnts = Counter(ngrams(ref, i + 1))
                for ng in hypcnts:
                    max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
            clipcnt = dict((ng, min(count, max_counts[ng])) \
                           for ng, count in hypcnts.items())
            clip_count[i] += sum(clipcnt.values())

        # accumulate r & c
        bestmatch = [1000, 1000]
        for ref in refs:
            if bestmatch[0] == 0: break
            diff = abs(len(ref) - len(hyp))
            if diff < bestmatch[0]:
                bestmatch[0] = diff
                bestmatch[1] = len(ref)
        r += bestmatch[1]
        c += len(hyp)
        if n == 1:
            break
    p0 = 1e-7
    bp = 1 if c > r else math.exp(1 - float(r) / float(c))
    p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
            for i in range(4)]
    s = math.fsum(w * math.log(p_n) \
                  for w, p_n in zip(weights, p_ns) if p_n)
    bleu = bp * math.exp(s)
    return bleu    
    
class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, old_hypothesis, old_corpus, n=1):
        file_names = old_hypothesis.keys()
        hypothesis = []
        corpus = []        
        for f in file_names:
            old_h = old_hypothesis[f]
            old_c = old_corpus[f]
            for h, c in zip(old_h, old_c):
                hypothesis.append([h])
                corpus.append([c])   
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]
        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            # Shawn's evaluation
            refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']
            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu
    
class Tokenizer(object):
    def __init__(self, vocab, ivocab, lower_case=True):
        super(Tokenizer, self).__init__()
        self.lower_case = lower_case
        self.ivocab = ivocab
        self.vocab = vocab
        
        self.vocab_len = len(self.vocab)

    def tokenize(self, sent):
        if self.lower_case:
            return sent.lower().split()
        else:
            return sent.split()

    def get_word_id(self, w, template=None):        
        if w in self.vocab:
            return self.vocab[w]
        else:
            return self.vocab[Constants.UNK_WORD]
        
    
    def get_word(self, k, template=None):
        k = str(k)
        return self.ivocab[k]
            
    def convert_tokens_to_ids(self, sent, template=None):
        return [self.get_word_id(w, template) for w in sent]

    def convert_id_to_tokens(self, word_ids, template_ids=None, remain_eos=False):
        if isinstance(word_ids, list):
            if remain_eos:
                return " ".join([self.get_word(wid, None) for wid in word_ids 
                                 if wid != Constants.PAD])
            else:
                return " ".join([self.get_word(wid, None) for wid in word_ids 
                                 if wid not in [Constants.PAD, Constants.EOS] ])                
        else:
            if remain_eos:
                return " ".join([self.get_word(wid.item(), None) for wid in word_ids 
                                 if wid != Constants.PAD])
            else:
                return " ".join([self.get_word(wid.item(), None) for wid in word_ids 
                                 if wid not in [Constants.PAD, Constants.EOS]])
            
    def convert_template(self, template_ids):
        return [self.get_word(wid) for wid in template_ids if wid != Constants.PAD]  
""""
def nondetokenize(d_p, d_r):
    UNK = "xxxxxxx"
    placeholder = json.load(open('data/placeholder.json'))
    dialog_id = 0
    for dialog, gt_dialog in zip(d_p, d_r):
        turn_id = 0
        for turn, gt_turn in zip(dialog, gt_dialog):
            kb = gt_turn['KB']
            bs = gt_turn['BS']
            acts = gt_turn['act']
            ref = gt_turn['sys_orig']
            
            def change_words(domain, word, acts, keys, act_keys, kb_cols):              
                for key, act_key, kb_col in zip(keys, act_keys, kb_cols):
                    if key in word:
                        
                        if "reference" in word:
                            for act_name in acts:
                                if domain in act_name and act_key in act_name and acts[act_name] != "?":
                                    new = acts[act_name].lower()
                                    return new
                            
                        if kb != "None" and kb_col in kb[0]:
                            new = kb[1][kb[0].index(kb_col)].lower()
                            return new
                return None
                
            words = turn.split(' ')
            for i in range(len(words)):
                word = words[i]
                if word in placeholder:
                    if "reference" in word:
                        done = False
                        for act_name in acts:
                            if "ref" in act_name:
                                words[i] = acts[act_name].lower()
                                done = True
                                break
                        if not done:
                            words[i]= UNK
                    else:
                        if "attraction" in word:
                            new = change_words("attraction", words[i], acts, ["address", "area", "name", "phone", "postcode", "pricerange"], 
                                            ["addr", "area", "name", "phone", "post", "price"],
                                            ["address", "area", "name", "phone", "postcode", "pricerange"])
                            if new:
                                words[i] = new
                        elif "hotel" in word:
                            new = change_words("hotel", words[i], acts, ["name", "phone", "address", "postcode", "pricerange", "area"], 
                                            ["name", "phone", "addr", "post", "price", "area"],
                                            ["name", "phone", "address", "postcode", "pricerange", "area"])
                            if new:
                                words[i] = new                            
                        elif "restaurant" in word:
                            new = change_words("restaurant", words[i], acts, ["name", "phone", "address", "postcode", "food", "pricerange", "area"], 
                                               ["name", "phone", "addr", "post", "food", "price", "area"],
                                               ["name", "phone", "address", "postcode", "food", "pricerange", "area"])
                            if new:
                                words[i] = new                        
                        elif "train" in word:
                            new = change_words("train", words[i], acts, ["trainid", "price"], ["id", "ticket"], ["trainID", "price"])
                            if new:
                                words[i] = new                        
                        elif "police" in word:
                            new = change_words("police", words[i], acts, ["name", "phone", "address", "postcode"], 
                                                ["name", "phone", "addr", "post"],
                                                ["name", "phone", "address", "postcode"])
                            if new:
                                words[i] = new                        
                        elif "hospital" in word:
                            new = change_words("hospital", words[i], acts, ["name", "phone", "address", "postcode", "department", "name"], 
                                               ["name", "phone", "address", "postcode", "department", "name"],
                                               ["name", "phone", "address", "postcode", "department", "name"])
                            if new:
                                words[i] = new                        
                        elif "taxi" in word:
                            new = change_words("taxi", words[i], acts, ["phone", "type"], ["phone", "car"], ["phone", "type"])
                            if new:
                                words[i] = new                        
                        elif "value_count" in word:
                            words[i] = "1"
                        
                        elif "value_time" in word:
                            words[i] = "1:00"
                        
                        elif "value_day" in word:
                            words[i] = "monday"
                            
                        elif "value_place" in word:
                            words[i] = "cambridge"
            
            new_words = " ".join(words)
            d_p[dialog_id][turn_id] = new_words
            turn_id += 1
        dialog_id += 1
"""
def nondetokenize(d_p, d_r):
    need_replace = 0
    success = 0
    for gt_dialog_info in d_r:
        file_name = gt_dialog_info['file']
        gt_dialog = gt_dialog_info['info']
        for turn_id in range(len(d_p[file_name])):
            act = gt_dialog[turn_id]['act']
            words = d_p[file_name][turn_id].split(' ')
            counter = {}
            for i in range(len(words)):
                if "[" in words[i] and "]" in words[i]:
                    need_replace += 1.
                    domain, slot = words[i].split('_')
                    domain = domain[1:].capitalize()
                    slot = slot[:-1].capitalize()
                    key = '-'.join((domain, slot))
                    flag = False
                    for intent in act:
                        _domain, _intent = intent.split('-')
                        if domain == _domain and _intent in ['Inform', 'Recommend', 'Offerbook']:
                            for values in act[intent]:
                                if (slot == values[0]) and ('none' != values[-1]) and ((key not in counter) or (counter[key] == int(values[1])-1)):
                                    words[i] = values[-1]
                                    counter[key] = int(values[1])
                                    flag = True
                                    success += 1.
                                    break
                            if flag:
                                break
            d_p[file_name][turn_id] = " ".join(words)
    success_rate = success / need_replace
    return success_rate
"""
class Templator(object):
    with open('data/placeholder.json') as f:
        fields = json.load(f)['field']    
    templates = {}
    for f in fields:
        if 'pricerange' in f:
            templates[f] = "its price is {}".format(f)
        elif 'type' in f:
            templates[f] = "it is of {} type".format(f)
        elif "address" in f:
            templates[f] = "its address is {}".format(f)
        elif "name" in f:
            templates[f] = "its name is {}".format(f)
        elif "postcode" in f:
            templates[f] = "its postcode is {}".format(f)
        elif "phone" in f:
            templates[f] = "its phone number is {}".format(f)
        elif "reference" in f:
            templates[f] = "its reference is {}".format(f)
        elif "area" in f:
            templates[f] = "it is located in {}".format(f)
        elif "arriveby" in f:
            templates[f] = "it arrives by {}".format(f)
        elif "departure" in f:
            templates[f] = "it departs at {}".format(f)
        elif "destination" in f:
            templates[f] = "its destination is at {}".format(f)
        elif "day" in f:
            templates[f] = "it is at the time of {}".format(f)
        elif "stars" in f:
            templates[f] = "it has {} stars".format(f)
        elif "department" in f:
            templates[f] = "its department is {}".format(f)
        elif "food" in f:
            templates[f] = "it provides {} food".format(f)
        elif "duration" in f:
            templates[f] = "it takes {} long".format(f)
        elif "leaveat" in f:
            templates[f] = "it leaves at {}".format(f)
        elif "trainid" in f:
            templates[f] = "its train id is {}".format(f)
        elif "price" in f:
            templates[f] = "its price is {}".format(f)
        elif "entrance" in f:
            templates[f] = "its fee is {}".format(f)
        elif "parking":
            templates[f] = {"yes":"it has parking", "no":"it does not have parking"}
        elif "internet":
            templates[f] = {"no":"it has internet", "no":"it does not have internet"}
            
    @staticmethod
    def source2tempalte(source):
        string = ""
        for k, v in source.items():
            if "_id]" not in k: 
                if k in Templator.templates:
                    if isinstance(Templator.templates[k], str):
                        string += Templator.templates[k] + " . "
                    else:
                        string += Templator.templates[k][v] + " . "
        return string        
"""     
            