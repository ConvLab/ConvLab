import json
import os

db_dir = "."
files = [os.path.join(db_dir, item) for item in os.listdir(db_dir) if 'db.json' in item]

def extract_domain_name(file):
    return file[2:].split('_db.json')[0]

db_dic = {}
for file in files:
    domain = extract_domain_name(file)
    data = json.load(open(file))
    domain_dic = {}
    print('domain: {}'.format(domain))
    if domain == 'taxi':
        assert isinstance(data, dict)
        for key, value_list in data.items():
            assert isinstance(value_list, list)
            key = key.split('_')[1].lower()
            if key in ['phone']:
                continue
            value_list = [item.lower() for item in value_list]
            domain_dic[key] = list(set(value_list))
    else:
        assert isinstance(data, list)
        for item in data:
            assert isinstance(item, dict)
            for key, value in item.items():
                if key in ['id', 'location', 'price', "phone"]:
                    continue
                key = key.lower()
                value = value.lower()
                if key not in domain_dic:
                    domain_dic[key] = []
                if value not in domain_dic[key]:
                    domain_dic[key].append(value)
    db_dic[domain] = domain_dic

json.dump(db_dic, open('./db_values.json', 'w+'), indent=4)
