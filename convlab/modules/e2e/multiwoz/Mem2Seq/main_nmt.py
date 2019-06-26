# Modified by Microsoft Corporation.
# Licensed under the MIT license.

import logging

from models.Mem2Seq_NMT import Mem2Seq
from tqdm import tqdm
from utils.config import *
from utils.utils_NMT import prepare_data_seq

train,lang, max_len, max_r = prepare_data_seq(batch_size = 32)

model = Mem2Seq(hidden_size= 100, max_len= max_len, 
                max_r= max_r, lang=lang, 
                path="",lr=0.001, n_layers=3, dropout=0.0)

avg_best = 0
for epoch in range(300):
    logging.info("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar: 
        model.train_batch(input_batches=data[0], 
                          input_lengths=data[1], 
                          target_batches=data[2], 
                          target_lengths=data[3], 
                          target_index=data[4], 
                          batch_size=len(data[1]),
                          clip= 10.0,
                          teacher_forcing_ratio=0.5,
                          reset=(i==0))

        pbar.set_description(model.print_loss())

    if((epoch+1) % 1 == 0):    
        bleu = model.evaluate(train,avg_best)
        model.scheduler.step(bleu)
        if(bleu >= avg_best):
            avg_best = bleu
            cnt=0
        else:
            cnt+=1

        if(cnt == 5): break
