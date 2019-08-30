import json
import os
import zipfile
from pprint import pprint
import re
from annotate import phrase_idx_utt, phrase_in_utt
from copy import deepcopy


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def write_zipped_json(filepath, filename, json_data):
    json.dump(json_data, open(filename, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(filename)
    # TODO: remove file
    # os.remove(filename)


def add_name_annotation(data, database):
    all_domain = ['attraction', 'restaurant', 'hotel', 'police']
    for session_id, session in data.items():
        active_domain = [k for k in all_domain if session['goal'][k]]
        for i, turn in enumerate(session['log']):
            if i%2==1:
                # Whether process system annotation?
                continue
            utt = turn['text']
            da = turn['dialog_act']
            span_info = turn['span_info']
            # copy_da = deepcopy(da)
            # copy_span = deepcopy(span_info)
            exist_name_idx = {x[2]: utt.lower().index(x[2].lower()) for x in span_info if x[1]=='Name' or x[1]=='Dest' or x[1]=='Depart'}
            name2idx = {}
            for domain in active_domain:
                name2idx[domain] = {}
                name_set = [x['name'] for x in database[domain]]
                for name in name_set:
                    if phrase_in_utt(name, utt):
                        idx = utt.lower().index(name.lower())
                        exist_name = None
                        for k,v in exist_name_idx.items():
                            if v == idx:
                                exist_name = k
                                break
                        if exist_name:
                            if exist_name.lower() in name_set:
                                continue
                            else:
                                for intent, svs in da.items():
                                    for s,v in svs:
                                        if s=='Name' and v==exist_name:
                                            da[intent].remove([s,v])
                                for x in span_info:
                                    if x[1]=='Name' and x[2]==exist_name:
                                        span_info.remove(x)
                                name2idx[domain][utt[idx:idx+len(name)]] = idx
                        else:
                            word_overlap = [x for x in exist_name_idx.keys() if name.lower() in x.lower() or x.lower() in name.lower()]
                            if word_overlap or not exist_name_idx:
                                for exist_name in word_overlap:
                                    for intent, svs in da.items():
                                        for s,v in svs:
                                            if s=='Name' and v==exist_name:
                                                da[intent].remove([s,v])
                                    for x in span_info:
                                        if x[1]=='Name' and x[2]==exist_name:
                                            span_info.remove(x)
                                name2idx[domain][utt[idx:idx+len(name)]] = idx
                            else:
                                continue
                for name, idx in name2idx[domain].items():
                    word_index_begin, word_index_end = phrase_idx_utt(name, utt)
                    span_info.append((domain.capitalize()+'-Inform', 'Name', name, word_index_begin, word_index_end))
                    da.setdefault(domain.capitalize()+'-Inform',[])
                    da[domain.capitalize()+'-Inform'].append(['Name', name])
            tmp_da = deepcopy(da)
            for intent in tmp_da:
                if not da[intent]:
                    da.pop(intent)
            # if da!=copy_da:
            #     print(utt)
            #     print(copy_da)
            #     print(data[session_id]['log'][i]['dialog_act'])
            #     print(copy_span)
            #     print(span_info)
            #     print()
    return data


if __name__ == '__main__':
    database = {
        'attraction': json.load(open('../db/attraction_db.json')),
        'restaurant': json.load(open('../db/restaurant_db.json')),
        'hotel': json.load(open('../db/hotel_db.json')),
        'train': json.load(open('../db/train_db.json')),
        'police': json.load(open('../db/police_db.json')),
        'hospital': json.load(open('../db/hospital_db.json')),
    }
    data = read_zipped_json('annotated_user_da_with_span_full.json.zip', 'annotated_user_da_with_span_full.json')
    add_name_annotation(data, database)
    write_zipped_json('annotated_user_da_with_span_full_patchName.json.zip', 'annotated_user_da_with_span_full_patchName.json', data)
