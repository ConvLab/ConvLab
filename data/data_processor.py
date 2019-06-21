# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
'''
from __future__ import print_function

import argparse
import json

from nltk.tokenize import sent_tokenize, word_tokenize

""" ====== NLG Part ====== """

def prepare_nlg_data(params):
    """ prepare nlg data """
    
    raw_file_path = params['raw_file_path']
    
    raw_data = json.load(open(raw_file_path, 'rb'))
    
    print(("#dialog", len(raw_data)))
    print("sessID\tMsgID\tMsgFrom\tText\tDialogAct")
    
    #f = open('./multiwoz/annotated_user_utts.txt', 'w')
    #f.write("sessID\tMsgID\tText\tDialogAct\n")
            
    for key in raw_data.keys()[0:]:
        dialog = raw_data[key]
        print('%s %d' % (key, len(dialog['log'])))
        
        for turn_id, turn in enumerate(dialog['log']):
            txt = turn['text']
            
            print("%d %s" % (turn_id, txt))
            
            dia_acts =  turn['dialog_act']
            #print('%d', len(dia_acts))
            
            if len(dia_acts) == 0 or len(dia_acts) > 1: continue
            
            dia_acts_str = []
            for dia_act in dia_acts:
                dia_act_str = ""
                dia_act_intent = dia_act
                dia_act_slot_vals = ""
                
                for slot_val in dia_acts[dia_act]:
                    slot = slot_val[0]
                    val = slot_val[1]
                    
                    if slot == 'none': continue
                    else: pass
                    
                    if val == "none" or val == "?":
                        dia_act_slot_vals += slot + ";"
                    else:
                        dia_act_slot_vals += slot + "=" + val.strip() + ";"
                    
                dia_acts_str.append(dia_act_str)
                
                #print('%s: %s' % (dia_act, dia_acts[dia_act]))
                
                if dia_act_slot_vals.endswith(";"): 
                    dia_act_slot_vals = dia_act_slot_vals[0:-1].strip()
                print('%s(%s)' % (dia_act, dia_act_slot_vals))
                
                #f.write(key+"\t"+str(turn_id)+"\t"+txt+"\t"+dia_act+"("+dia_act_slot_vals+")\n")
    
    #f.close()
    
def prepare_data(params):
    """ prepare data """
    
    raw_file_path = params['raw_file_path']
    
    raw_data = json.load(open(raw_file_path, 'rb'))
    
    print(("#dialog", len(raw_data)))
    print("sessID\tMsgID\tMsgFrom\tText\tDialogAct")
    
    f = open('./multiwoz/annotated_user_utts_all.txt', 'w')
    f.write("sessID\tMsgID\tText\tDialogAct\n")
    
    unaligned = 0
    total = 0
            
    for key in raw_data.keys()[0:]:
        dialog = raw_data[key]
        #print('%s %d' % (key, len(dialog['log'])))
        
        for turn_id, turn in enumerate(dialog['log']):
            total += 1
            txt = turn['text']
            sentences = sent_tokenize(txt) #txt.split('.')
            
            dia_acts =  turn['dialog_act']
            #print('%d %d', len(sentence), len(dia_acts))
            
            if len(dia_acts) == 0 or (len(dia_acts) > 1 and len(sentences) != len(dia_acts)):
            #if len(dia_acts) == 0 or len(dia_acts) > 1: 
                
                unaligned += 1
                
                #print("%d %s" % (turn_id, txt))
                #print(dia_acts)
                continue
            
            for dia_act_id, dia_act in enumerate(list(dia_acts.keys())):
                dia_act_intent = dia_act
                dia_act_slot_vals = ""
                
                txt_str = sentences[dia_act_id]
                txt_str = txt_str.replace('\t', " ")
                txt_str = txt_str.replace('\n', "")
                
                for slot_val in dia_acts[dia_act]:
                    slot = slot_val[0]
                    val = slot_val[1]
                    
                    if slot == 'none': continue
                    
                    if val == "none" or val == "?":
                        dia_act_slot_vals += slot + ";"
                    else:
                        dia_act_slot_vals += slot + "=" + val.strip() + ";"
                
                #print('%s: %s' % (dia_act, dia_acts[dia_act]))
                
                if dia_act_slot_vals.endswith(";"): 
                    dia_act_slot_vals = dia_act_slot_vals[0:-1].strip()
                    
                #print('%s' % (txt_str))
                #print('%s(%s)' % (dia_act, dia_act_slot_vals))
                
                f.write(key+"\t"+str(turn_id)+"\t"+txt_str+"\t"+dia_act+"("+dia_act_slot_vals+")\n")
    
    print('unaligned/total: %d/%d' % (unaligned, total))
    f.close()
    
def prepare_dia_acts_slots(params):
    """ prepare dialog acts and slots """
    
    raw_file_path = params['dia_act_slot']
    
    file = open(raw_file_path, 'r')
    lines = [line.strip().strip('\n').strip('\r') for line in file]
    
    f = open('./multiwoz/slot_set.txt', 'w')
    
    slot_set = set()        
    for l in lines:
        arr = l.split('\t')
        if len(arr) > 1: slot_set.add(arr[1].strip())
    
    for s in slot_set:
        print(s)
    f.close()
    
  
""" ====== NLU Part ====== """
    
def generate_bio_from_raw_data(params):
    """  cmd = 2: read the raw data, and generate BIO file """
    
    wfile = open("./multiwoz/annoted_bio_all_tst.txt", "w")
    
    file = open(params['txt_dia_act_file'], 'r')
    lines = [line.strip().strip('\n').strip('\r') for line in file]
    
    print(('lines', len(lines)))
    
    for l_id, l in enumerate(lines[1:]):
        fields = l.split('\t')
        
        sessID = fields[0].strip()
        msID = fields[1].strip()
        msTxt = fields[2].strip()
        dia_acts = fields[3].strip()
        
        dia_act = parse_str_to_diaact(dia_acts)
        
        #if dia_act['intent'] == "inform": print sessID, msID, dia_acts
        
        print(l_id, sessID, msID, dia_acts, dia_act)
        
        new_str, bio_str, intent = parse_str_to_bio(msTxt, dia_act)
        
        #print dia_acts, intent
        #print new_str
        #print bio_str
                
        wfile.write(new_str + '\t' + bio_str + '\n')
        
    wfile.close()   

""" ====== Generate NLU Part ====== """

def generate_nlu_bio_from_data(params):
    """ prepare nlg data """
    
    raw_file_path = params['raw_file_path']
    raw_data = json.load(open(raw_file_path, 'rb'))
    
    print(("#dialog", len(raw_data)))
    print("sessID\tMsgID\tText\tDialogAct")
    
    wfile = open("./multiwoz/annoted_bio_all.txt", "w")
            
    for key in raw_data.keys()[0:]:
        dialog = raw_data[key]
        print('%s %d' % (key, len(dialog['log'])))
        
        for turn_id, turn in enumerate(dialog['log']):
            txt = turn['text']
            txt = txt.replace('\t', " ")
            txt = txt.replace('\n', "")
            
            dia_acts = turn['dialog_act']
            
            print("%d %s %s" % (turn_id, txt, dia_acts))
            
            intents, new_str, bio_str = parse_s_to_bio(txt, dia_acts)
            print('%s, %s' % (new_str, bio_str))
            
            wfile.write(new_str + '\t' + bio_str + '\n')
        
    wfile.close()
    
def parse_s_to_bio(s, dia_acts):
    """ parse s with dia_acts """
    
    intents = []
    slot_vals = {}
    for intent in dia_acts:
        dia_act = dia_acts[intent]
        
        for s_v in dia_act:        
            if s_v[0] == "none" or s_v[1] == 'none' or s_v[1] == '?':
                if s_v[0] != "none":
                    intents.append(intent+"+"+s_v[0])
                else: 
                    intents.append(intent)
                continue
            
            if 'inform' in intent or 'Inform' in intent:
                new_slot = intent + "+" + s_v[0]
                slot_vals[new_slot] = s_v[1]
                continue
            
            if s_v[0] != "none" and s_v[1] != "none" and s_v[1] != '?':
                new_slot = intent + "+" + s_v[0]
                slot_vals[new_slot] = s_v[1]
                
    if len(intents) == 0: intents.append('N/A')
    
    w_arr, bio_arr = parse_slotvals(s, slot_vals)
    bio_arr[-1] = ','.join(intents)
    new_str = ' '.join(w_arr)
    bio_str = ' '.join(bio_arr)
    
    return intents, new_str, bio_str

def parse_slotvals(s, slot_vals):
    """ parse slot-vals """
    
    new_str = 'BOS ' + s + ' EOS'
    new_str = new_str.lower()
    w_arr = word_tokenize(new_str) #new_str.split(' ')
    bio_arr = ['O'] * len(w_arr)
    
    left_index = 0
    for slot in slot_vals:
        slot_val = slot_vals[slot].lower()
        
        if len(slot_vals[slot]) == 0: continue
        
        str_left_index = new_str.find(slot_val, 0)
        if str_left_index == -1: 
            str_left_index = new_str.find(slot_val.split(' ')[0], 0)
        
        if str_left_index == -1: continue
            
        left_index = len(s[0:str_left_index].split(' '))
        
        #print(slot_val, str_left_index, left_index, len(w_arr), len(slot_val.split(' ')))
        
        range_len = min(len(slot_val.split(' ')), len(w_arr)-left_index)
        for index in range(range_len):
            bio_arr[left_index+index] = ("B-" + slot) if index == 0 else ("I-" + slot)
    
    return w_arr, bio_arr

def select_single_intent(params):
    """ select single intent utterances """
    
    file = open(params['txt_dia_act_file'], 'r')
    lines = [line.strip().strip('\n').strip('\r') for line in file]
    
    print(('lines', len(lines)))
    
    wfile = open("./multiwoz/annoted_bio_all_20k.txt", "w")
    
    N = 20000
    line = 0
    for l_id, l in enumerate(lines[0:]):
        fields = l.split('\t')
        
        if line > N: break
        
        #if (len(fields[0].split(" ")) != len(fields[1].split(" "))): print l_id, l
        
        intents = fields[1].split(',')
        if len(intents) == 1:
            wfile.write(l + '\n')
            line += 1
    wfile.close()
    


def parse_str_to_diaact(s):
    """ parse str to dia_act """
    
    dia_act = {}
    
    intent = ""
    slot_val_s = ""
        
    if s.find('(') > 0 and s.find(')') > 0:
        intent = s[0: s.find('(')].strip(' ').lower()
        slot_val_s = s[s.find('(')+1: -1].strip(' ') #slot-value pairs
        
        dia_act['intent'] = intent
        
        #if len(annot) == 0: continue #confirm_question()

    if len(slot_val_s) > 0:
        # slot-pair values: slot[val] = id
        annot_segs = slot_val_s.split(';') #slot-value pairs
        sent_slot_vals = {} # slot-pair real value

        for annot_seg in annot_segs:
            annot_seg = annot_seg.strip(' ')
            annot_slot = annot_seg
            if annot_seg.find('=') > 0:
                left_index = 0
                #if annot_seg.find('||') > 0: left_index = annot_seg.find('||')+2
                    
                annot_slot = annot_seg[left_index:annot_seg.find('=', left_index)].strip(' ') #annot_seg.split('=')[0].strip(' ')                
                annot_val = annot_seg[annot_seg.find('=')+1:].strip(' ') #annot_seg.split('=')[1].strip(' ')
            else: #requested
                annot_val = 'UNK' # for request
                if annot_slot == 'taskcomplete': annot_val = 'FINISH'
                
            if annot_slot == 'mc_list':
                #left_index = 0
                #if annot_seg.find('{')> 0: left_index = annot_seg.find('{') + 1
                #annot_slot = annot_seg[left_index:annot_seg.find('=', left_index)] #annot_seg.split('=')[0].strip(' ')
                #annot_val = annot_seg[annot_seg.find('=', left_index)+1:]
                continue

            # slot may have multiple values
            sent_slot_vals[annot_slot] = []

            if annot_val.startswith('{') and annot_val.endswith('}'): # multiple-choice or result={}
                annot_val = annot_val[1:-1].strip(' ')

                if annot_slot == 'result': # result={slot=value}
                    result_annot_seg_arr = annot_val.strip(' ').split('&')
                    if len(annot_val.strip(' '))> 0:
                        for result_annot_seg_item in result_annot_seg_arr:
                            result_annot_seg_arr = result_annot_seg_item.strip(' ').split('=')

                            result_annot_seg_slot = result_annot_seg_arr[0]
                            result_annot_seg_slot_val = result_annot_seg_arr[1]
                            sent_slot_vals[annot_slot].append({result_annot_seg_slot:result_annot_seg_slot_val})
                    else: # result={}
                        pass
                else: # multi-choice or mc_list
                    annot_val_arr = annot_val.split('#')
                    for annot_val_item in annot_val_arr:
                        sent_slot_vals[annot_slot].append(annot_val_item.strip(' '))
            else: # single choice
                sent_slot_vals[annot_slot].append(annot_val)
        
        dia_act['slot_vals'] = sent_slot_vals    
    else: # no slot-value pairs
        dia_act['slot_vals'] = {} 

    return dia_act

def parse_str_to_bio(str, dia_act):
    """ parse str to BIO format """
    
    intent = parse_intent(dia_act)
    w_arr, bio_arr = parse_slots(str, dia_act)
    bio_arr[-1] = intent
    
    return ' '.join(w_arr), ' '.join(bio_arr), intent

def parse_intent(dia_act):
    """ parse intent """
    
    intent_word = dia_act['intent']
    intent = intent_word
    
    if intent_word == 'inform':
        if 'taskcomplete' in dia_act['slot_vals'].keys():
            intent += '+taskcomplete'
    elif 'request' in intent_word: # request intent
        for slot in dia_act['slot_vals'].keys():
            if 'UNK' in dia_act['slot_vals'][slot]:
                intent += '+' + slot
    else:
        pass
    
    return intent

def parse_slots(str, dia_act):
    """ format BIO """
    
    new_str = 'BOS ' + str + ' EOS'
    new_str = new_str.lower()
    w_arr = new_str.split(' ')
    bio_arr = ['O'] * len(w_arr)
    
    left_index = 0
    for slot in dia_act['slot_vals'].keys():
        if len(dia_act['slot_vals'][slot]) == 0: continue
        
        slot_val = dia_act['slot_vals'][slot][0].lower()
        if slot_val == 'unk' or slot_val == 'finish': continue
        
        str_left_index = new_str.find(slot_val, 0)
        if str_left_index == -1: 
            str_left_index = new_str.find(slot_val.split(' ')[0], 0)
        
        if str_left_index == -1: continue
            
        left_index = len(str[0:str_left_index].split(' '))
        
        print((str_left_index, left_index, len(w_arr), len(slot_val.split(' '))))
        
        range_len = min(len(slot_val.split(' ')), len(w_arr)-left_index)
        for index in range(range_len):
            bio_arr[left_index+index] = ("B-" + slot) if index == 0 else ("I-" + slot)
    
    return w_arr, bio_arr


""" ====== Turn Pairs ====== """

def build_turn_pairs(params):
    """ construct turn pairs """
    
    raw_file_path = params['raw_file_path']
    raw_data = json.load(open(raw_file_path, 'rb'))
    
    print(("#dialog", len(raw_data)))
    
    dialogs = {}
    for dialog_key in raw_data:
        dialog = raw_data[dialog_key]
        

def sample_N_dialogues(params):
    """ sample N dialogues """
    
    raw_file_path = params['raw_file_path']
    raw_data = json.load(open(raw_file_path, 'rb'))
    
    print(("#dialog", len(raw_data)))
    
    N = 1000
    dialogs = {}
    for dialog_key in list(raw_data.keys()): #[0:N]:
        dialogs[dialog_key] = raw_data[dialog_key]
        
    with open('./multiwoz/annotated_user_da_with_span_'+str(N)+'sample.json', 'w') as fp:
        json.dump(dialogs, fp)
    


def prepare_query_res_pairs(params):
    """ prepare query-response raw pairs """
    
    raw_file_path = params['raw_file_path']
    raw_data = json.load(open(raw_file_path, 'rb'))
    
    print(("#dialog", len(raw_data)))
    print("sessID\tMsgID\tText\tDialogAct")
    
    f = open('./multiwoz/annotated_qr_pairs.txt', 'w')
    f.write("sessID\tMsgID\tText\tUsrAct\tSysAct\n")
    
    unaligned = 0
    total = 0
    pairs = []
            
    for key in raw_data.keys()[0:]:
        dialog = raw_data[key]
        print('%s %d' % (key, len(dialog['log'])))
        
        turn_id = 0
        while turn_id < len(dialog['log']):
            pair = {}
            
            turn = dialog['log'][turn_id]
            txt = turn['text']
            txt = txt.replace('\t', " ")
            txt = txt.replace('\n', "")
            
            dia_acts = turn['dialog_act']
            usr_acts = []
            for dia_act_id, dia_act in enumerate(list(dia_acts.keys())):
                dia_act_intent = dia_act
                dia_act_slots = []
                
                for slot_val in dia_acts[dia_act]:
                    slot = slot_val[0]
                    val = slot_val[1]
                    
                    if slot == 'none': continue
                    
                    if val == "none" or val == "?":
                        dia_act_intent += "_" + slot
                    else:
                        dia_act_slots.append(slot)
                
                usr_acts.append('%s(%s)' % (dia_act, ';'.join(dia_act_slots)))
                #print('%s(%s)' % (dia_act, ';'.join(dia_act_slots)))
            
            system_turn = dialog['log'][turn_id+1]
            sys_dia_acts = system_turn['dialog_act']
            sys_acts = []
            for dia_act_id, dia_act in enumerate(list(sys_dia_acts.keys())):
                dia_act_intent = dia_act
                dia_act_slots = []
                
                for slot_val in sys_dia_acts[dia_act]:
                    slot = slot_val[0]
                    val = slot_val[1]
                    
                    if slot == 'none': continue
                    
                    if val == "none" or val == "?":
                        dia_act_intent += "_" + slot
                    else:
                        dia_act_slots.append(slot)
                
                sys_acts.append('%s(%s)' % (dia_act, ';'.join(dia_act_slots)))
                #print('%s(%s)' % (dia_act, ';'.join(dia_act_slots)))
            
            pair['sessID'] = key
            pair['turnID'] = turn_id
            pair['txt'] = txt
            pair['usr_acts'] = usr_acts
            pair['sys_acts'] = sys_acts
            
            print('%s(%s)' % (usr_acts, sys_acts))
            f.write(key+"\t"+str(turn_id)+"\t"+txt+"\t"+','.join(usr_acts)+"\t"+','.join(sys_acts)+"\n")
            turn_id += 2
            
def prepare_u_s_pairs(params):
    """ prepare query-response pairs """
    
    raw_file_path = params['raw_file_path']
    raw_data = json.load(open(raw_file_path, 'rb'))
    
    print(("#dialog", len(raw_data)))
    print("sessID\tMsgID\tText\tDialogAct")
    
    f = open('./multiwoz/annotated_us_pairs.txt', 'w')
    
    total = 0
    pairs = []
            
    for key in raw_data.keys()[0:]:
        dialog = raw_data[key]
        print('%s %d' % (key, len(dialog['log'])))
        
        turn_id = 0
        while turn_id < len(dialog['log']):
            pair = {}
            
            turn = dialog['log'][turn_id]
            txt = turn['text']
            txt = txt.replace('\t', " ")
            txt = txt.replace('\n', "")
            
            dia_acts = turn['dialog_act']
            usr_acts = []
            for dia_act_id, dia_act in enumerate(list(dia_acts.keys())):
                dia_act_intent = dia_act
                dia_act_slots = []
                
                for slot_val in dia_acts[dia_act]:
                    slot = slot_val[0]
                    val = slot_val[1]
                    
                    if slot == 'none': continue
                    
                    if val == "none" or val == "?":
                        dia_act_intent += "_" + slot
                    else:
                        dia_act_slots.append(slot)
                
                usr_acts.append('%s(%s)' % (dia_act, ';'.join(dia_act_slots)))
                #print('%s(%s)' % (dia_act, ';'.join(dia_act_slots)))
            
            system_turn = dialog['log'][turn_id+1]
            sys_dia_acts = system_turn['dialog_act']
            sys_acts = []
            for dia_act_id, dia_act in enumerate(list(sys_dia_acts.keys())):
                dia_act_intent = dia_act
                dia_act_slots = []
                
                for slot_val in sys_dia_acts[dia_act]:
                    slot = slot_val[0]
                    val = slot_val[1]
                    
                    if slot == 'none': continue
                    
                    if val == "none" or val == "?":
                        dia_act_intent += "_" + slot
                    else:
                        dia_act_slots.append(slot)
                
                sys_acts.append('%s(%s)' % (dia_act, ';'.join(dia_act_slots)))
                #print('%s(%s)' % (dia_act, ';'.join(dia_act_slots)))
            
            pair['sessID'] = key
            pair['turnID'] = turn_id
            pair['txt'] = txt
            pair['usr_acts'] = usr_acts
            pair['sys_acts'] = sys_acts
            
            print('%s(%s)' % (usr_acts, sys_acts))
            f.write(txt+"\t"+','.join(sys_acts)+"\n")
            turn_id += 2



    
def main(params):
    
    cmd = params['cmd']
    
    if cmd == 0: # generate nlg data
        prepare_nlg_data(params)
    elif cmd == 1: # generate dia_act, slot sets
        prepare_dia_acts_slots(params)
    elif cmd == 2: # generate NLU BIO
        generate_bio_from_raw_data(params)
    elif cmd == 3: # construct turn pairs
        build_turn_pairs(params)
    elif cmd == 4: # sample N dialogues
        sample_N_dialogues(params)
    elif cmd == 5: # prepare data
        prepare_data(params)
    elif cmd == 6: # create sl policy training data
        prepare_query_res_pairs(params)
    elif cmd == 7:
        prepare_u_s_pairs(params)
    elif cmd == 8: # generate nlu bio
        generate_nlu_bio_from_data(params)
    elif cmd == 9: # select single intent
        select_single_intent(params)
    elif cmd == 10:
        pass
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cmd', dest='cmd', type=int, default=8, help='cmd')
    
    parser.add_argument('--raw_file_path', dest='raw_file_path', type=str, default='./multiwoz/annotated_user_da_with_span_100sample.json', help='path to data file')
    parser.add_argument('--dia_act_slot', dest='dia_act_slot', type=str, default='./multiwoz/dialog_act_slot.txt', help='path to data file')
    
    parser.add_argument('--txt_dia_act_file', dest='txt_dia_act_file', type=str, default='./multiwoz/annotated_user_utts_18k.txt', help='path to data file')
    
    args = parser.parse_args()
    params = vars(args)
    
    print('Setup Parameters: ')
    print(json.dumps(params, indent=2))

    main(params)