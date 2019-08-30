import json
import os
import re
import sys
import zipfile

import spacy

proj_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../.."))
sys.path.insert(0, proj_path)

REF_USR_DA = {
    'Attraction': {
        'area': 'Area', 'type': 'Type', 'name': 'Name',
        'entrance fee': 'Fee', 'address': 'Addr',
        'postcode': 'Post', 'phone': 'Phone'
    },
    'Hospital': {
        'department': 'Department', 'address': 'Addr', 'postcode': 'Post',
        'phone': 'Phone'
    },
    'Hotel': {
        'type': 'Type', 'parking': 'Parking', 'pricerange': 'Price',
        'internet': 'Internet', 'area': 'Area', 'stars': 'Stars',
        'name': 'Name', 'stay': 'Stay', 'day': 'Day', 'people': 'People',
        'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone'
    },
    'Police': {
        'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone', 'name': 'Name'
    },
    'Restaurant': {
        'food': 'Food', 'pricerange': 'Price', 'area': 'Area',
        'name': 'Name', 'time': 'Time', 'day': 'Day', 'people': 'People',
        'phone': 'Phone', 'postcode': 'Post', 'address': 'Addr'
    },
    'Taxi': {
        'leaveAt': 'Leave', 'destination': 'Dest', 'departure': 'Depart', 'arriveBy': 'Arrive',
        'car type': 'Car', 'phone': 'Phone'
    },
    'Train': {
        'destination': 'Dest', 'day': 'Day', 'arriveBy': 'Arrive',
        'departure': 'Depart', 'leaveAt': 'Leave', 'people': 'People',
        'duration': 'Time', 'price': 'Ticket', 'trainID': 'Id'
    }
}

REF_SYS_DA = {
    'Attraction': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Fee': "entrance fee", 'Name': "name", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Type': "type",
        'none': None, 'Open': None
    },
    'Hospital': {
        'Department': 'department', 'Addr': 'address', 'Post': 'postcode',
        'Phone': 'phone', 'none': None
    },
    'Booking': {
        'Day': 'day', 'Name': 'name', 'People': 'people',
        'Ref': 'ref', 'Stay': 'stay', 'Time': 'time',
        'none': None
    },
    'Hotel': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Internet': "internet", 'Name': "name", 'Parking': "parking",
        'Phone': "phone", 'Post': "postcode", 'Price': "pricerange",
        'Ref': "ref", 'Stars': "stars", 'Type': "type",
        'none': None
    },
    'Restaurant': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Name': "name", 'Food': "food", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Ref': "ref",
        'none': None
    },
    'Taxi': {
        'Arrive': "arriveBy", 'Car': "car type", 'Depart': "departure",
        'Dest': "destination", 'Leave': "leaveAt", 'Phone': "phone",
        'none': None
    },
    'Train': {
        'Arrive': "arriveBy", 'Choice': "choice", 'Day': "day",
        'Depart': "departure", 'Dest': "destination", 'Id': "trainID",
        'Leave': "leaveAt", 'People': "people", 'Ref': "ref",
        'Time': "duration", 'none': None, 'Ticket': 'price',
    },
    'Police': {
        'Addr': "address", 'Post': "postcode", 'Phone': "phone"
    },
}

init_belief_state = {
        'taxi': {'book': {'booked': []}, 'semi': {'leaveAt': '', 'destination': '', 'departure': '', 'arriveBy': ''}},
        'police': {'book': {'booked': []}, 'semi': {}},
        'restaurant': {'book': {'booked': [], 'time': '', 'day': '', 'people': ''},
                       'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}},
        'hospital': {'book': {'booked': []}, 'semi': {'department': ''}},
        'hotel': {'book': {'booked': [], 'stay': '', 'day': '', 'people': ''},
                  'semi': {'name': '', 'area': '', 'parking': '', 'pricerange': '', 'stars': '', 'internet': '',
                           'type': ''}},
        'attraction': {'book': {'booked': []}, 'semi': {'type': '', 'name': '', 'area': ''}},
        'train': {'book': {'booked': [], 'people': '', 'ticket': ''},
                  'semi': {'leaveAt': '', 'destination': '', 'day': '', 'arriveBy': '', 'departure': ''}}
}
digit2word = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
}
word2digit = {v:k for k,v in digit2word.items()}


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def un_zip(file_name):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)
    dir_name = '.'
    if os.path.isdir(dir_name):
        pass
    else:
        os.mkdir(dir_name)
    for names in zip_file.namelist():
        zip_file.extract(names, dir_name)
    zip_file.close()


def tokenize(data, process_text=True, process_da=True, process_ref=True):
    print('Begin tokenization:')
    print('='*50)
    nlp = spacy.load('en_core_web_sm')
    cnt = 0
    for no, session in data.items():
        cnt += 1
        if cnt % 1000 == 0:
            print('[%d|%d]' % (cnt,len(data)))
        for turn in session['log']:
            if process_text:
                doc = nlp(turn['text'])
                turn['text'] = ' '.join([token.text for token in doc]).strip()
            if process_da:
                for da, svs in turn['dialog_act'].items():
                    for i in range(len(svs)):
                        if svs[i][0] == 'Ref' and not process_ref:
                            continue
                        svs[i][1] = ' '.join([token.text for token in nlp(svs[i][1])]).strip()
    print('=' * 50)
    print('Finish tokenization')


def dict_diff(dict1,dict2):
    # compare two dict
    # two exceptions:
    # 1) 'bus' domain unuse
    # 2) 'ticket' and 'people' attr for 'train-book' domain may be missing
    diff_dict = {}
    for k,v2 in dict2.items():
        if k in dict1:
            assert isinstance(v2, type(dict1[k]))
            v1 = dict1[k]
            if v1 != v2:
                if not isinstance(v2, type({})):
                    diff_dict[k] = v2
                else:
                    if dict_diff(v1,v2)!={}:
                        diff_dict[k] = dict_diff(v1,v2)
        else:
            if k!='bus':
                assert k=='people'
                # people attribute for train domain
                if v2!='':
                    diff_dict[k] = v2
    return diff_dict


def phrase_in_utt(phrase, utt):
    phrase_low = phrase.lower()
    utt_low = utt.lower()
    phrases = [phrase_low]
    if phrase_low in digit2word:
        phrases.append(digit2word[phrase_low])
    elif phrase_low in word2digit:
        phrases.append(word2digit[phrase_low])
    else:
        if ' '+phrase_low in utt_low or utt_low.startswith(phrase_low):
            return True
        else:
            return False

    for w in phrases:
        if utt_low.startswith(w) or utt_low.endswith(w):
            return True
        elif ' '+w+' ' in utt_low:
            return True
    return False


def phrase_idx_utt(phrase, utt):
    phrase_low = phrase.lower()
    utt_low = utt.lower()
    phrases = [phrase_low]
    if phrase_low in digit2word:
        phrases.append(digit2word[phrase_low])
    elif phrase_low in word2digit:
        phrases.append(word2digit[phrase_low])
    else:
        if ' '+phrase_low in utt_low or utt_low.startswith(phrase_low):
            return get_idx(phrase_low, utt_low)
        else:
            return None
    for w in phrases:
        if utt_low.startswith(w) or utt_low.endswith(w):
            return get_idx(w, utt_low)
        elif ' '+w+' ' in utt_low:
            return get_idx(' '+w+' ', utt_low)
        # elif w+'-star' in utt_low:
        #     return get_idx(w, utt_low)
    return None


def get_idx(phrase, utt):
    char_index_begin = utt.index(phrase)
    char_index_end = char_index_begin + len(phrase)
    word_index_begin = len(utt[:char_index_begin].split())
    word_index_end = len(utt[:char_index_end].split()) - 1
    return word_index_begin, word_index_end


def annotate_user_da(data):
    print('Begin user da annotation:')
    print('=' * 50)
    # empty initial state

    domains = ['taxi', 'police', 'hospital', 'hotel', 'attraction', 'train', 'restaurant']

    nlp = spacy.load('en_core_web_sm')

    for no, session in data.items():
        user_das = []
        user_goal = session['goal']
        for i in range(1, len(session['log']), 2):
            prev_state = init_belief_state if i == 1 else session['log'][i - 2]['metadata']
            next_state = session['log'][i]['metadata']
            prev_utterance = '' if i == 1 else session['log'][i - 2]['text']
            next_utterance = session['log'][i]['text']
            # doc = nlp(session['log'][i - 1]['text'])
            # user_utterance = ' '.join([token.text for token in doc]).strip()
            user_utterance = session['log'][i - 1]['text']
            if i == 1:
                prev_da = {}
            else:
                prev_da = session['log'][i - 2]['dialog_act']
            next_da = session['log'][i]['dialog_act']
            diff_table = dict_diff(prev_state, next_state)

            # user da annotate, Inform
            da = {}
            for domain in domains:
                if len(user_goal[domain]) != 0:
                    da[domain.capitalize() + '-Inform'] = []
                    for slot in REF_USR_DA[domain.capitalize()].keys():
                        value_state = ''
                        if domain in diff_table:
                            for subtable in diff_table[domain].values():
                                if slot in subtable and subtable[slot] != 'not mentioned':
                                    value_state = subtable[slot]
                                    # state for that slot change
                        # value_state = value_state.lower()
                        if value_state != '':
                            value_state = ' '.join([token.text for token in nlp(value_state)]).strip()

                        value_goal = ''
                        if slot in user_goal[domain]['info']:
                            value_goal = user_goal[domain]['info'][slot]
                        elif 'book' in user_goal[domain] and slot in user_goal[domain]['book']:
                            value_goal = user_goal[domain]['book'][slot]
                        # value_goal = value_goal.lower()
                        if value_goal != '':
                            value_goal = ' '.join([token.text for token in nlp(value_goal)]).strip()

                        # slot-value appear in goal
                        slot_in_da = REF_USR_DA[domain.capitalize()][slot]

                        if value_state != '':
                            # state change
                            if phrase_in_utt(value_state, user_utterance):
                                # value in user utterance
                                da[domain.capitalize() + '-Inform'].append([slot_in_da, value_state])
                            elif phrase_in_utt(slot, user_utterance):
                                # slot in user utterance
                                da[domain.capitalize() + '-Inform'].append([slot_in_da, value_state])
                            elif slot == 'stars' and (
                                    phrase_in_utt(value_state + '-star', user_utterance) or
                                    phrase_in_utt(value_state + '-stars', user_utterance)):
                                da[domain.capitalize() + '-Inform'].append([slot_in_da, value_state])
                            elif slot == 'people' and phrase_in_utt('one person', user_utterance):
                                # keyword 'person' for people
                                da[domain.capitalize() + '-Inform'].append([slot_in_da, "1"])
                            elif slot == 'stay' and phrase_in_utt('night', user_utterance):
                                # keyword 'night' for stay
                                da[domain.capitalize() + '-Inform'].append([slot_in_da, value_state])
                            elif slot == 'internet' and phrase_in_utt('wifi', user_utterance):
                                # alias 'wifi' for internet
                                da[domain.capitalize() + '-Inform'].append([slot_in_da, value_state])
                            elif slot == 'pricerange' and phrase_in_utt('price', user_utterance):
                                # alias 'price' for pricerange
                                da[domain.capitalize() + '-Inform'].append([slot_in_da, value_state])
                            elif slot == 'arriveBy' and phrase_in_utt('arrive', user_utterance):
                                # alias 'arrive' for arriveBy
                                if value_state == value_goal:
                                    da[domain.capitalize() + '-Inform'].append([slot_in_da, value_state])
                            elif slot == 'leaveAt' and phrase_in_utt('leave', user_utterance):
                                # alias 'leave' for leaveAt
                                if value_state != value_goal:
                                    da[domain.capitalize() + '-Inform'].append([slot_in_da, value_state])
                            elif domain.capitalize() + '-Request' in prev_da and [slot_in_da, '?'] in prev_da[
                                domain.capitalize() + '-Request']:
                                # answer system request
                                da[domain.capitalize() + '-Inform'].append([slot_in_da, value_state])
                            elif value_goal != '' and phrase_in_utt(value_goal, user_utterance):
                                # wrong state update
                                da[domain.capitalize() + '-Inform'].append([slot_in_da, value_state])

                        elif value_goal != '':
                            # state don't change but value is in goal
                            if value_goal == 'yes' or value_goal == 'no':
                                # binary value
                                if '?' in user_utterance:
                                    # not for acknowledgement
                                    if phrase_in_utt(slot, user_utterance):
                                        da[domain.capitalize() + '-Inform'].append([slot_in_da, value_goal])
                                    elif slot == 'internet' and phrase_in_utt('wifi', user_utterance):
                                        da[domain.capitalize() + '-Inform'].append([slot_in_da, value_goal])
                            elif value_goal.isdigit():
                                # digital value
                                if phrase_in_utt(value_goal, user_utterance):
                                    if slot == 'stars':
                                        if phrase_in_utt('star ', user_utterance) or \
                                                phrase_in_utt('stars ',user_utterance) or \
                                                user_utterance.lower().endswith('star') or \
                                                user_utterance.lower().endswith('stars') or \
                                                '-star' in user_utterance.lower():
                                            da[domain.capitalize() + '-Inform'].append([slot_in_da, value_goal])
                                    elif slot == 'stay':
                                        if phrase_in_utt('stay', user_utterance) or \
                                                phrase_in_utt('night', user_utterance) or \
                                                phrase_in_utt('nights', user_utterance):
                                            da[domain.capitalize() + '-Inform'].append([slot_in_da, value_goal])
                                    elif slot == 'people':
                                        if phrase_in_utt('people', user_utterance) or phrase_in_utt('person', user_utterance):
                                            if phrase_in_utt(domain, user_utterance) or phrase_in_utt(domain, prev_utterance):
                                                da[domain.capitalize() + '-Inform'].append([slot_in_da, value_goal])
                                        elif phrase_in_utt('ticket', user_utterance):
                                            if domain == 'train':
                                                da[domain.capitalize() + '-Inform'].append([slot_in_da, value_goal])
                                    else:
                                        assert 0
                            elif phrase_in_utt(value_goal, user_utterance):
                                # string value
                                if phrase_in_utt(domain, user_utterance) or phrase_in_utt(domain, prev_utterance):
                                    da[domain.capitalize() + '-Inform'].append([slot_in_da, value_goal])

                    if len(da[domain.capitalize() + '-Inform']) == 0:
                        da.pop(domain.capitalize() + '-Inform')

            # Request
            for domain in domains:
                if len(user_goal[domain]) != 0:
                    da[domain.capitalize() + '-Request'] = []
                    for slot in REF_USR_DA[domain.capitalize()].keys():
                        # for each possible request in goal
                        slot_in_da = REF_USR_DA[domain.capitalize()][slot]
                        if 'reqt' in user_goal[domain] and slot in user_goal[domain]['reqt']:
                            # if actually in request goal
                            if phrase_in_utt(slot, user_utterance):
                                da[domain.capitalize() + '-Request'].append([slot_in_da, '?'])
                            elif slot == 'internet' and phrase_in_utt('wifi', user_utterance):
                                da[domain.capitalize() + '-Request'].append([slot_in_da, '?'])
                            elif slot == 'postcode' and phrase_in_utt('post', user_utterance):
                                da[domain.capitalize() + '-Request'].append([slot_in_da, '?'])
                            elif slot == 'pricerange' and phrase_in_utt('price', user_utterance):
                                da[domain.capitalize() + '-Request'].append([slot_in_da, '?'])
                            elif slot == 'trainID' and phrase_in_utt('id', user_utterance):
                                da[domain.capitalize() + '-Request'].append([slot_in_da, '?'])
                            elif slot == 'arriveBy' and phrase_in_utt('arrive', user_utterance):
                                if phrase_in_utt('when', user_utterance) or phrase_in_utt('what time', user_utterance):
                                    da[domain.capitalize() + '-Request'].append([slot_in_da, '?'])
                            elif slot == 'leaveAt' and phrase_in_utt('leave', user_utterance):
                                if phrase_in_utt('when', user_utterance) or phrase_in_utt('what time', user_utterance):
                                    da[domain.capitalize() + '-Request'].append([slot_in_da, '?'])
                            elif slot == 'duration' and phrase_in_utt('travel time', user_utterance):
                                da[domain.capitalize() + '-Request'].append([slot_in_da, '?'])
                            elif slot == 'arriveBy' and phrase_in_utt('arrival time', user_utterance):
                                da[domain.capitalize() + '-Request'].append([slot_in_da, '?'])
                            elif slot == 'leaveAt' and phrase_in_utt('departure time', user_utterance):
                                da[domain.capitalize() + '-Request'].append([slot_in_da, '?'])

                    # for request not in goal
                    if phrase_in_utt('reference', user_utterance) or phrase_in_utt('ref', user_utterance):
                        if 'book' in user_goal[domain] and len(user_goal[domain]['book']) > 0:
                            da[domain.capitalize() + '-Request'].append(['Ref', '?'])

                    if len(da[domain.capitalize() + '-Request']) == 0:
                        da.pop(domain.capitalize() + '-Request')

            # Clarify the domain of request slot (for address, postcode, area, phone,...)
            slot2domain = {}
            for domain_da, svs in da.items():
                if 'Request' in domain_da:
                    for (s, v) in svs:
                        slot2domain.setdefault(s, [])
                        slot2domain[s].append(domain_da.split('-')[0])
            #         print(slot2domain)
            for s, d in slot2domain.items():
                if len(d) > 1:
                    # several request for same slot
                    # note that in data no slot alias appear twice
                    if len(re.findall(s, user_utterance)) <= 1:
                        # for slot don't appear twice
                        system_ack = []
                        for each in d:
                            if each + '-Inform' in next_da:
                                if s in map(lambda x: x[0], next_da[each + '-Inform']):
                                    # system Inform the value of slot:
                                    system_ack.append(each)
                            elif each + '-Recommend' in next_da:
                                if s in map(lambda x: x[0], next_da[each + '-Recommend']):
                                    # system Recommend the value of slot:
                                    system_ack.append(each)
                            elif each + '-OfferBooked' in next_da:
                                if s in map(lambda x: x[0], next_da[each + '-OfferBooked']):
                                    # system Recommend the value of slot:
                                    system_ack.append(each)

                        if len(system_ack) == 0:
                            # not informed or recommended by system, abort. 227 samples
                            for each in d:
                                for request_slot_value in da[each + '-Request']:
                                    if s == request_slot_value[0]:
                                        da[each + '-Request'].remove(request_slot_value)
                                if len(da[each + '-Request']) == 0:
                                    da.pop(each + '-Request')
                        elif len(system_ack) == 1:
                            # one of domain informed or recommended by system. 1441 samples
                            for each in d:
                                if each in system_ack:
                                    continue
                                for request_slot_value in da[each + '-Request']:
                                    if s == request_slot_value[0]:
                                        da[each + '-Request'].remove(request_slot_value)
                                if len(da[each + '-Request']) == 0:
                                    da.pop(each + '-Request')
                        elif len(system_ack) == 2:
                            # two of domain informed or recommended by system. 3 samples
                            pass
                        else:
                            # no >2 sample
                            assert 0

            # General
            if len(da) == 0:
                for domain in domains:
                    if phrase_in_utt(domain, user_utterance):
                        da.setdefault(domain.capitalize() + '-Inform', [])
                        da[domain.capitalize() + '-Inform'].append(['none', 'none'])
            if len(da) == 0:
                if phrase_in_utt('bye', user_utterance):
                    da['general-bye'] = [['none', 'none']]
                elif phrase_in_utt('thank', user_utterance):
                    da['general-thank'] = [['none', 'none']]
                elif sum([1 if phrase_in_utt(x, user_utterance) else 0 for x in ['hello', 'hi']]) > 0:
                    da['general-greet'] = [['none', 'none']]

            user_das.append(da)
            if no=='MUL0800':
                print(user_utterance)
                print(da)
                print(next_da)
                print(diff_table)

        for j, user_da in enumerate(user_das):
            session['log'][j * 2]['dialog_act'] = user_das[j]

    print('=' * 50)
    print('End user da annotation')


def annotate_sys_da(data, database):
    print('Begin system da annotation:')
    print('=' * 50)
    police_val2slot = {
        "Parkside Police Station": "name",
        "Parkside , Cambridge": "address",
        "01223358966": "phone",
        "CB11JG": "postcode"
    }
    police_slots = set(police_val2slot.values())
    police_vals = set(police_val2slot.keys())
    print('police slot:', police_slots)

    hospital_val2slot = {}
    hospital_slot2val = {}
    for d in database['hospital']:
        for k, v in d.items():
            if k != 'id':
                hospital_val2slot[v] = k
                hospital_slot2val.setdefault(k,[])
                hospital_slot2val[k].append(v)
    hospital_slot2val['phone'].append('01223245151')
    hospital_slot2val['address']=['Hills Rd , Cambridge']
    hospital_slot2val['postcode'] = ['CB20QQ']
    hospital_val2slot['01223245151'] = 'phone'
    hospital_val2slot['Hills Rd , Cambridge'] = 'address'
    hospital_val2slot['CB20QQ'] = 'postcode'
    hospital_slots = set(database['hospital'][0].keys())-{'id'}
    hospital_slots = hospital_slots | {'address'}
    hospital_vals = set(hospital_val2slot.keys())
    print('hospital slot:', hospital_slots)
    for no, session in data.items():
        for i in range(0, len(session['log']), 1):
            das = session['log'][i]['dialog_act']
            utterance = session['log'][i]['text']
            more_da = {}

            # Police-Inform
            more_da['Police-Inform'] = []
            for val in police_vals:
                if phrase_in_utt(val, utterance):
                    slot_in_da = REF_USR_DA['Police'][police_val2slot[val]]
                    more_da['Police-Inform'].append([slot_in_da, val])
            if len(more_da['Police-Inform']) > 0:
                das['Police-Inform'] = more_da['Police-Inform']

            # Hospital-Inform
            more_da['Hospital-Inform'] = []
            for val in hospital_vals:
                if phrase_in_utt(val, utterance):
                    slot_in_da = REF_USR_DA['Hospital'][hospital_val2slot[val]]
                    more_da['Hospital-Inform'].append([slot_in_da, val])
            if len(more_da['Hospital-Inform']) > 0:
                if 'Hospital-Inform' not in das:
                    das['Hospital-Inform'] = more_da['Hospital-Inform']

            # Police-Request already done in user da annotation (system don't ask)
            # more_da['Police-Request'] = []
            # if phrase_in_utt('police', utterance):
            #     for val, slot in police_val2slot.items():
            #         if phrase_in_utt(slot, utterance) \
            #                 and not phrase_in_utt(val, utterance) \
            #                 and 'Police-Inform' not in das \
            #                 and 'Police-Request' not in das:
            #             print(utterance)
            #             print(das)

            # Hospital-Request:department for system
            more_da['Hospital-Request'] = []
            if phrase_in_utt('hospital', utterance):
                slot = 'department'
                if phrase_in_utt(slot, utterance):
                    for val in hospital_slot2val['department']:
                        if phrase_in_utt(val, utterance):
                            break
                    else:
                        if i%2==1:
                            das['Hospital-Request'] = [[REF_USR_DA['Hospital'][slot], '?']]

            for da, svs in das.items():
                for j, (s, v) in enumerate(svs):
                    if s == 'Ref':
                        real_v = ''.join(v.split())
                        if v in session['log'][i]['text']:
                            session['log'][i]['text'] = session['log'][i]['text'].replace(v, real_v)
                        if real_v+'.' in session['log'][i]['text']:
                            session['log'][i]['text'] = session['log'][i]['text'].replace(real_v+'.', real_v+' .')
                        svs[j][1] = real_v
    print('=' * 50)
    print('End system da annotation')


def annotate_span(data):
    da2anno = {'Inform', 'Select', 'Recommend', 'NoOffer', 'NoBook', 'OfferBook', 'OfferBooked', 'Book'}
    total_da = 0
    anno_da = 0
    for no, session in data.items():
        for i in range(0, len(session['log']), 1):
            das = session['log'][i]['dialog_act']
            utterance = session['log'][i]['text']
            span_info = []
            for da, svs in das.items():
                da_flag = False
                if sum([1 if x == da.split('-')[1] else 0 for x in da2anno]) > 0:
                    da_flag = True
                if da_flag:
                    for s, v in svs:
                        if s != 'Internet' and s != 'Parking' and s != 'none':
                            is_annotated = False
                            if phrase_in_utt(v, utterance):
                                # mentioned explicitly
                                is_annotated = True
                                word_index_begin, word_index_end = phrase_idx_utt(v, utterance)
                                span_info.append((da, s, v, word_index_begin, word_index_end))
                            elif s == 'Stars':
                                pattern = ''
                                if phrase_in_utt(v+'-star', utterance):
                                    pattern = v+'-star'
                                elif phrase_in_utt(v+'-stars', utterance):
                                    pattern = v+'-stars'
                                if pattern:
                                    is_annotated = True
                                    word_index_begin, word_index_end = phrase_idx_utt(pattern, utterance)
                                    span_info.append((da, s, v, word_index_begin, word_index_end))
                            elif phrase_in_utt('same', utterance) and phrase_in_utt(s, utterance):
                                # coreference-'same'
                                if phrase_in_utt('same ' + s, utterance):
                                    is_annotated = True
                                    assert len(s.split()) == 1
                                    word_index_begin, word_index_end = phrase_idx_utt('same ' + s, utterance)
                                    span_info.append((da, s, v, word_index_begin, word_index_end))
                                elif s == 'People':
                                    is_annotated = True
                                    if phrase_in_utt('same group of people', utterance):
                                        pattern = 'same group of people'
                                    elif phrase_in_utt('same number of people', utterance):
                                        pattern = 'same number of people'
                                    elif phrase_in_utt('same amount of people', utterance):
                                        pattern = 'same amount of people'
                                    elif phrase_in_utt('same quantity of people', utterance):
                                        pattern = 'same quantity of people'
                                    else:
                                        assert 0
                                    word_index_begin, word_index_end = phrase_idx_utt(pattern, utterance)
                                    span_info.append((da, s, v, word_index_begin, word_index_end))
                                else:
                                    word_index_begin, word_index_end = phrase_idx_utt('same', utterance)
                                    shift = len(utterance[:utterance.lower().index(s.lower())].split()) - word_index_begin
                                    if 0 < shift <= 3:
                                        is_annotated = True
                                        span_info.append((da, s, v, word_index_begin, word_index_begin + shift))
                            elif 'care' in v:
                                # value: don't care
                                key_phrases = ["not particular", "no particular", "any ", "not really", "do n't matter",
                                               "do n't care", "do not care", "do n't really care", "do nt care",
                                               "does n't matter", "does nt matter", "do n't have a preference",
                                               "do not have a preference", "does n't really matter", "does not matter"]
                                for key_phrase in key_phrases:
                                    if phrase_in_utt(key_phrase, utterance):
                                        word_index_begin, word_index_end = phrase_idx_utt(key_phrase, utterance)
                                        span_info.append((da, s, v, word_index_begin, word_index_end))
                                        is_annotated = True
                                        break
                            elif ':' in v and ':' in utterance.lower():
                                # time value
                                char_index_begin = utterance.lower().index(':')
                                word_index_begin = len(utterance[:char_index_begin].split()) - 1
                                if utterance.lower().split()[word_index_begin - 1] == 'after' or \
                                        utterance.lower().split()[word_index_begin - 1] == 'before':
                                    span_info.append((da, s, v, word_index_begin - 1, word_index_begin))
                                else:
                                    span_info.append((da, s, v, word_index_begin, word_index_begin))
                                is_annotated = True
                            elif v == 'centre' and phrase_in_utt('center', utterance):
                                word_index_begin, word_index_end = phrase_idx_utt('center', utterance)
                                span_info.append((da, s, v, word_index_begin, word_index_begin))
                                is_annotated = True
                            if is_annotated:
                                anno_da += 1
                            total_da += 1
            session['log'][i]['span_info'] = span_info


def post_process_span(data):
    for no, session in data.items():
        for i in range(0, len(session['log']), 1):
            das = session['log'][i]['dialog_act']
            utterance = session['log'][i]['text']
            span_info = session['log'][i]['span_info']
            start_end_pos = dict()
            for act, slot, value, start, end in span_info:
                if (start,end) in start_end_pos:
                    start_end_pos[(start,end)].append([act, slot, value, start, end])
                else:
                    start_end_pos[(start,end)] = [[act, slot, value, start, end]]
            for start_end in start_end_pos:
                if len(start_end_pos[start_end]) > 1:
                    value = [x[2] for x in start_end_pos[start_end]]
                    if len(set(value))>1:
                        # print(utterance)
                        # print(start_end_pos[start_end])
                        for ele in start_end_pos[start_end]:
                            v = ele[2]
                            if utterance.startswith(v+' '):
                                new_span = get_idx(v+' ',utterance)
                            elif ' '+v+' ' in utterance:
                                new_span = get_idx(' ' + v + ' ', utterance)
                            else:
                                new_span = None
                            if new_span:
                                ele[3], ele[4] = new_span
                        # print(start_end_pos[start_end])
                    else:
                        # one value
                        for ele in start_end_pos[start_end]:
                            slot = ele[1]
                            v = ele[2]
                            if slot == 'People':
                                pattern = ''
                                if phrase_in_utt('people', utterance):
                                    pattern = 'people'
                                elif phrase_in_utt('person', utterance):
                                    pattern = 'person'
                                if pattern:
                                    slot_span = phrase_idx_utt(pattern, utterance)
                                    v_set = [v]
                                    if v in digit2word:
                                        v_set.append(digit2word[v])
                                    elif v in word2digit:
                                        v_set.append(word2digit[v])
                                    if utterance.split()[slot_span[0]-1] in v_set:
                                        ele[3], ele[4] = slot_span[0]-1, slot_span[1]-1

                            elif slot == 'Stay':
                                pattern = ''
                                if phrase_in_utt('night', utterance):
                                    pattern = 'night'
                                elif phrase_in_utt('nights', utterance):
                                    pattern = 'nights'
                                if pattern:
                                    slot_span = phrase_idx_utt(pattern, utterance)
                                    v_set = [v]
                                    if v in digit2word:
                                        v_set.append(digit2word[v])
                                    elif v in word2digit:
                                        v_set.append(word2digit[v])
                                    if utterance.split()[slot_span[0]-1] in v_set:
                                        ele[3], ele[4] = slot_span[0]-1, slot_span[1]-1
                        # print(start_end_pos[start_end])
            new_span_info = [x for y in start_end_pos.values() for x in y]
            session['log'][i]['span_info'] = new_span_info




if __name__ == '__main__':
    un_zip('MULTIWOZ2.zip')
    dir_name = 'MULTIWOZ2 2/'
    database = {
        'attraction': read_json(dir_name + 'attraction_db.json'),
        'hotel': read_json(dir_name + 'hotel_db.json'),
        'restaurant': read_json(dir_name + 'restaurant_db.json'),
        # 'taxi': read_json(dir_name + 'taxi_db.json'),
        'train': read_json(dir_name + 'train_db.json'),
        'police': read_json(dir_name + 'police_db.json'),
        'hospital': read_json(dir_name + 'hospital_db.json'),
    }
    dialog_acts = read_json(dir_name+'dialogue_acts.json')
    data = read_json(dir_name+'data.json')
    sessions_key = list(map(lambda x: x.split('.')[0], data.keys()))
    all_data = {}
    for session in sessions_key:
        all_data[session] = data[session + '.json']
        if len(all_data[session]['log']) - len(dialog_acts[session]) * 2 > 1:
            # some annotation are incomplete
            all_data.pop(session)
            continue
        for i, turn in enumerate(all_data[session]['log']):
            if i % 2 == 0:
                turn['dialog_act'] = {}
            else:
                da = dialog_acts[session]['%d' % ((i + 1) / 2)]
                if da == 'No Annotation':
                    turn['dialog_act'] = {}
                else:
                    turn['dialog_act'] = da
    print('dataset size: %d' % len(all_data))
    # all_data = dict(list(all_data.items())[-100:])
    tokenize(all_data, process_text=True, process_da=True, process_ref=True)
    annotate_user_da(all_data)
    annotate_sys_da(all_data, database)
    tokenize(all_data, process_text=False, process_da=True, process_ref=False)
    annotate_span(all_data)
    # archive = zipfile.ZipFile('annotated_user_da_with_span_full.json.zip', 'r')
    # all_data = json.load(archive.open('annotated_user_da_with_span_full.json'))
    # annotate_span(all_data)
    post_process_span(all_data)
    # # json.dump(all_data, open('test.json', 'w'), indent=4)
    json.dump(all_data, open('annotated_user_da_with_span_full.json', 'w'), indent=4)
    with zipfile.ZipFile('annotated_user_da_with_span_full.json.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write('annotated_user_da_with_span_full.json')

    os.remove('annotated_user_da_with_span_full.json')
