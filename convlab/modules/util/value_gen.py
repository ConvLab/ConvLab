import os
import json


db_dir = '../../data/multiwoz/db'

dbs = os.listdir(db_dir)
dbs = [os.path.join(db_dir, item) for item in dbs]

value_dict = {}

ignore_slot = ['id', 'location', 'phone']

for db in dbs:
    db_file = db.split('/')[-1]
    domain_name = db_file.split('_db.json')[0]
    # print(domain_name)
    data = json.load(open(db))
    # print(len(data))
    domain_dict = {}
    # ignore
    # all: id
    # attraction:
    print('domain: {}'.format(domain_name))
    if domain_name != 'taxi':
        for item in data:
            for k, v in item.items():
                assert type(k) is str
                if k in ignore_slot:
                    continue
                else:
                    if k in domain_dict:
                        if type(v) is str:
                            domain_dict[k].add(v)
                        elif type(v) is dict:
                            for _, sub_v in v.items():
                                domain_dict[k].add(sub_v)
                        else:
                            raise Exception('unexpected value type: {}'.format(type(v)))
                    else:
                        domain_dict[k] = set()
                        if type(v) is str:
                            domain_dict[k].add(v)
                        elif type(v) is dict:
                            for _, sub_v in v.items():
                                domain_dict[k].add(sub_v)
                        else:
                            raise Exception('unexpected value type: {}'.format(type(v)))
    else:
        for slot, values in data.items():
            assert type(slot) is str
            slot = slot.split('_')[1]
            if slot is 'phone': continue
            domain_dict[slot] = values
    new_domain_dict = {}
    for slot, v_set in domain_dict.items():
        new_domain_dict[slot] = list(v_set)

    for k, v_list in new_domain_dict.items():
        print('\t', k, len(v_list))
        for v in v_list:
            json.dumps(v, indent=4)
        json.dumps(k, indent=4)
        json.dumps(v_list, indent=4)

    value_dict[domain_name] = new_domain_dict

json.dump(value_dict, open('../../data/value_set.json', 'w+'), indent=4)
# json.dumps(value_dict, indent=4)



