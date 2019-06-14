import json
import os
import zipfile

archive = zipfile.ZipFile('annotation/annotated_user_da_with_span_full.json.zip', 'r')
dataset = json.load(archive.open('annotated_user_da_with_span_full.json'))
val_list = set(line.strip()[:-5] for line in open('annotation/MULTIWOZ2 2/valListFile.json').readlines())
test_list = set(line.strip()[:-5] for line in open('annotation/MULTIWOZ2 2/testListFile.json').readlines())

train_dials = {}
val_dials = {}
test_dials = {}

for dial_name in dataset:
    if dial_name in val_list:
        val_dials[dial_name] = dataset[dial_name]
    elif dial_name in test_list:
        test_dials[dial_name] = dataset[dial_name]
    else:
        train_dials[dial_name] = dataset[dial_name]

print('train/val/test size: {}/{}/{}'.format(len(train_dials),len(val_dials),len(test_dials)))
open('trainListFile','w').writelines([x+'\n' for x in train_dials.keys()])
open('valListFile','w').writelines([x+'\n' for x in val_dials.keys()])
open('testListFile','w').writelines([x+'\n' for x in test_dials.keys()])

json.dump(train_dials, open('train.json', 'w'), indent=2)
with zipfile.ZipFile('train.json.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('train.json')
json.dump(val_dials, open('val.json', 'w'), indent=2)
with zipfile.ZipFile('val.json.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('val.json')
json.dump(test_dials, open('test.json', 'w'), indent=2)
with zipfile.ZipFile('test.json.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('test.json')

os.remove('train.json')
os.remove('val.json')
os.remove('test.json')

