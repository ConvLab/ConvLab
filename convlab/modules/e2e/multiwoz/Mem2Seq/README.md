# Mem2Seq 

**Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems** (ACL 2018). Andrea Madotto, Chien-Sheng Wu, Pascale Fung. Accepted at ***ACL 2018***. [[PDF]](https://aclanthology.coli.uni-saarland.de/papers/P18-1136/p18-1136) in ACL anthology. [Andrea Madotto](http://andreamad8.github.io/) and [Chien-Sheng Wu](https://jasonwu0731.github.io/) contribute equally at this work.  

This code has been written using Pytorch 0.3, soon we will update the code to Pytorch 0.4.

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex are listed below:

<pre>
@InProceedings{P18-1136,
  author = 	"Madotto, Andrea
		and Wu, Chien-Sheng
		and Fung, Pascale",
  title = 	"Mem2Seq: Effectively Incorporating Knowledge Bases into End-to-End Task-Oriented Dialog Systems",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1468--1478",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/P18-1136"
}
</pre>

## Mem2Seq in pytorch

In this repository we implemented Mem2Seq and several baseline in pytorch (Version 0.3). To make the code more reusable we diveded each model in a separated files (obivuosly there is a large code overlap). In the folder models you can find the following:

- ***Mem2Seq***: Memory to Sequence (Our model)
- ***Seq2Seq***: Vanilla seq2seq model with no attention (enc_vanilla)
- ***+Attn***: Luong attention attention model
- ***Ptr-Unk***: combination between Bahdanau attention and Pointer Networks ([Point to UNK words](http://www.aclweb.org/anthology/P16-1014)) 

All of these file share the same structure, which is: a class that builds an encoder and a decoder, and provide training and validation methods (all inside the class).

## Import data

Under the utils folder, we have the script to import and batch the data for each dataset. 

## Basic example

Mem2Seq can be considered as a general sequence to sequence model with the ability to address external memories. We prepared a very basic implementation (including data preprocessing and model) for a English to France translation task. Obviusly there is not much to copy from the input in this small corpus, so it is just to show how the model works in a general sequence to sequence task. Run:

```console
❱❱❱ python3 main_nmt.py
```

This version uses a flat memory instead of triple as described in the paper. 

## Train a model for task-oriented dialog datasets

We created  `main_train.py` to train models. You can see there is a notation, `globals()[args['decoder']]`, it is converting a string into a fuction. So to train a model you can run:
Mem2Seq bAbI t1-t6:

```console
❱❱❱ python3 main_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.2 -dec=Mem2Seq -bsz=8 -ds=babi -t=1 
❱❱❱ python3 main_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.2 -dec=VanillaSeqToSeq -bsz=8 -ds=babi -t=1
❱❱❱ python3 main_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.2 -dec=LuongSeqToSeq -bsz=8 -ds=babi -t=1
❱❱❱ python3 main_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.2 -dec=PTRUNK -bsz=8 -ds=babi -t=1
```

or Mem2Seq In-Car

```console
❱❱❱ python3 main_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.2 -dec=Mem2Seq -bsz=8 -ds=kvr -t=
❱❱❱ python3 main_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.2 -dec=VanillaSeqToSeq -bsz=8 -ds=kvr -t=
❱❱❱ python3 main_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.2 -dec=LuongSeqToSeq -bsz=8 -ds=kvr -t=
❱❱❱ python3 main_train.py -lr=0.001 -layer=1 -hdd=128 -dr=0.2 -dec=PTRUNK -bsz=8 -ds=kvr -t=
```

the option you can choose are:

- `-t` this is task dependent. 1-6 for bAbI and nothing for In-Car
- `-ds` choose which dataset to use (babi and kvr)
- `-dec` to choose the model. The option are: Mem2Seq, VanillaSeqToSeq, LuongSeqToSeq, PTRUNK
- `-hdd` hidden state size of the two rnn
- `-bsz` batch size
- `-lr` learning rate
- `-dr` dropout rate
- `-layer` number of stacked rnn layers, or number of hops for Mem2Seq



While training, the model with the best validation is saved. If you want to reuse a model add `-path=path_name_model` to the function call. The model is evaluated by using per responce accuracy, WER, F1 and BLEU.

## Notes

For hyper-parameter search of Mem2Seq, our suggestions are:

- Try to use a higher dropout rate (dr >= 0.2) and larger hidden size (hdd>=256) to get better performance when training with small hop (H<=3). 
- While training Mem2Seq with larger hops (H>3), it may perform better with smaller hidden size (hdd<256) and higher dropout rate.
- Since there are some variances between runs, so it's better to run several times or run different seeds to get the best performance.

## What's new? (Gaoxin)

Please download [data, trained model & results](https://drive.google.com/open?id=1Z3gCaiyILhy8_3PCmlfXb26awwW8bq2S) and unzip the file here.

- I have updated the code to pytorch **1.0**
- Support **multiwoz** dataset (see *data/MultiWOZ/\*.txt*) for the **Mem2Seq** method (other baselines are not implemented yet)
```console
❱❱❱ python3 main_train.py -lr=0.001 -layer=1 -hdd=256 -dr=0.2 -dec=Mem2Seq -bsz=16 -ds=woz -t=
❱❱❱ python3 main_test.py -dec=Mem2Seq -bsz=16 -ds=woz -path=save/mem2seq-WOZ/[saved model dir]/
```

- The model are saved at *save/mem2seq-WOZ* directory, results are provided in the *pairs.txt*
- Interact with the agent using command line (command END/RESET to end/reset the current dialog session)
```console
❱❱❱ python3 main_interact.py -dec=Mem2Seq -ds=woz -path=save/mem2seq-WOZ/[saved model dir]/
```


