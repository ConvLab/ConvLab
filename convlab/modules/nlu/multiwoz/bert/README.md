# BERTNLU on multiwoz

Based on pre-trained bert, BERTNLU use a linear layer for slot tagging and another linear layer for intent classification. Dialog acts are split into two groups, depending on whether the value is in the utterance. 

- For those dialog acts that the value appears in the utterance, they are translated to BIO tags. For example, `"Find me a cheap hotel"`, its dialog act is `{"Hotel-Inform":[["Price", "cheap"]]}`, and translated tag sequence is `["O", "O", "O", "B-Hotel-Inform+Price", "O"]`. A linear layer takes pre-trained bert word embeddings as input and classify the tag label.
- For each of the other dialog acts, such as `(Hotel-Request, Address, ?)`, another linear layer takes pre-trained bert embeddings of `[CLS]` as input and do the binary classification.

## Usage

Determine which data you want to use: if **mode**='usr', use user utterances to train; if **mode**='sys', use system utterances to train; if **mode**='all', use both user and system utterances to train.

#### Preprocess data

On `bert/multiwoz` dir:

```sh
$ python preprocess.py [mode]
```

output processed data on `data/[mode]_data/` dir.

#### Train a model

On `bert` dir:

```sh
$ python train.py --config_path multiwoz/configs/multiwoz_[mode].json
```

The model will be saved on `output/[mode]/bestcheckpoint.tar`. Also, it will be zipped as `output/[mode]/bert_multiwoz_[mode].zip`. 

#### Evaluate

On `bert/multiwoz` dir:

```sh
$ python evaluate.py [mode]
```

#### Predict

In `nlu.py` , the `BERTNLU` class inherits the NLU interface and adapts to multiwoz dataset. Example usage:

```python
from convlab.modules.nlu import BERTNLU

model = BERTNLU(mode, model_file=PATH_TO_ZIPPED_MODEL_OR_MODEL_URL)
dialog_act = model.predict(utterance)
```

## Data

We use the multiwoz data (`data/multiwoz/[train|val|test].json.zip`).

## Performance

`mode` determines the data we use: if mode=`usr`, use user utterances to train; if mode=`sys`, use system utterances to train; if mode=`all`, use both user and system utterances to train.

We evaluate the precision/recall/f1 of predicted dialog act.

| mode | Precision | Recall | F1    |
| ---- | --------- | ------ | ----- |
| usr  | 72.55     | 76.33  | 74.40 |
| sys  | 69.45     | 72.59  | 70.99 |
| all  | 67.71     | 71.92  | 69.75 |

## References

```
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={4171--4186},
  year={2019}
}
```

