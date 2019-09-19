import os
import logging
import argparse
from tqdm import tqdm

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

USE_CUDA = True
MAX_LENGTH = 10

args = {}
args['load_embedding'] = True
args["fix_embedding"] = False
if args["load_embedding"]:
    args["hidden"] = 400
    print("[Warning] Using hidden size = 400 for pretrained word embedding (300 + 100)...")
# if args["fix_embedding"]:
#     args["addName"] += "FixEmb"
# if args["except_domain"] != "":
#     args["addName"] += "Except"+args["except_domain"]
# if args["only_domain"] != "":
#     args["addName"] += "Only"+args["only_domain"]


