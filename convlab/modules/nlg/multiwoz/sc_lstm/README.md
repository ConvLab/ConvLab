# nlg-sclstm-multiwoz

Semantically-conditioned LSTM (SC-LSTM) is an NLG model that generates natural linguistically varied responses based on a deep, semantically controlled LSTM architecture. The code derives from [github](https://github.com/andy194673/nlg-sclstm-multiwoz). We modify it to support user NLG. The original paper can be found at [ACL Anthology](https://aclweb.org/anthology/papers/D/D15/D15-1199/)

## Run the code

unzip [rar](https://drive.google.com/open?id=1cKotyFbff6VkPtJrpqDakiFjiXV34eyr) here

TRAIN
```bash
$ PYTHONPATH=../../../../.. python3 run_woz.py  --mode=train --model_path=sclstm.pt --n_layer=1 --lr=0.005 > sclstm.log
```

TEST

```bash
$ PYTHONPATH=../../../../.. python3 run_woz.py --mode=test --model_path=sclstm.pt --n_layer=1 --beam_size=10 > sclstm.res
```

Calculate BLEU

```bash
$ PYTHONPATH=../../../../.. python3 bleu.py --res_file=sclstm.res
```

Set *user* to use user NLG，e.g.
```bash
$ PYTHONPATH=../../../../.. python3 run_woz.py  --mode=train --model_path=sclstm_usr.pt --n_layer=1 --lr=0.005 --user True > sclstm_usr.log
```

## Data

We use the multiwoz data (./resource/\*, ./resource_usr/\*).

## Reference

```
@inproceedings{wen2015semantically,
  title={Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems},
  author={Wen, Tsung-Hsien and Gasic, Milica and Mrk{\v{s}}i{\'c}, Nikola and Su, Pei-Hao and Vandyke, David and Young, Steve},
  booktitle={Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing},
  pages={1711--1721},
  year={2015}
}
```
