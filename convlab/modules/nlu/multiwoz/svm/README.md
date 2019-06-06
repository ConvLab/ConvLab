# SVMNLU

SVMNLU build a classifier for each semantic tuple (intent-slot-value) based on n-gram features. It's first proposed by [Mairesse et al. (2009)](http://mairesse.s3.amazonaws.com/research/papers/icassp09-final.pdf). We adapt the implementation from [pydial](https://bitbucket.org/dialoguesystems/pydial/src/master/semi/CNetTrain/).

## Example usage

#### Preprocess data

```sh
$ python preprocess.py
```

output processed data on `corpora/data` dir.

#### Train a model

```sh
$ PYTHONPATH=../../../../.. python train.py config/multiwoz.cfg
```

The model will be saved on `model/multiwoz.pkl`. Also, it will be zipped as `model/svm_multiwoz.zip`. You can download this from `https://convlab.blob.core.windows.net/models/svm_multiwoz.zip` too.

#### Evaluate

In the parent directory:

```sh
$ PYTHONPATH=../../../.. python evaluate.py SVMNLU
```

The result on `data/multiwoz/test.json.zip`:

```
Model SVMNLU on 1000 session 14744 sentences:
         Precision: 75.44
         Recall: 43.29
         F1: 55.01
```

If you want to perform end-to-end evaluation with trained model, you can set `model_file` param of your ConvLab spec file to `"https://convlab.blob.core.windows.net/models/svm_multiwoz.zip"`. It will download the trained model and place it under `model` directory automatically.

## Data

We use the multiwoz data (data/multiwoz/[train|val|test].json.zip).

## References

```
@inproceedings{mairesse2009spoken,
  title={Spoken language understanding from unaligned data using discriminative classification models},
  author={Mairesse, Fran{\c{c}}ois and Gasic, Milica and Jurcicek, Filip and Keizer, Simon and Thomson, Blaise and Yu, Kai and Young, Steve},
  booktitle={2009 IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={4749--4752},
  year={2009},
  organization={IEEE}
}
@InProceedings{ultes2017pydial,
  author    = {Ultes, Stefan  and  Rojas Barahona, Lina M.  and  Su, Pei-Hao  and  Vandyke, David  and  Kim, Dongho  and  Casanueva, I\~{n}igo  and  Budzianowski, Pawe{\l}  and  Mrk\v{s}i\'{c}, Nikola  and  Wen, Tsung-Hsien  and  Gasic, Milica  and  Young, Steve},
  title     = {{PyDial: A Multi-domain Statistical Dialogue System Toolkit}},
  booktitle = {Proceedings of ACL 2017, System Demonstrations},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {73--78},
  url       = {http://aclweb.org/anthology/P17-4013}
}
```

