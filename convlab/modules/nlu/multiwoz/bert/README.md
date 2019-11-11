# BERTNLU on multiwoz

Based on pre-trained bert, BERTNLU use a linear layer for slot tagging and another linear layer for intent classification. Dialog acts are split into two groups, depending on whether the value is in the utterance. 

- For those dialog acts that the value appears in the utterance, they are translated to BIO tags. For example, `"Find me a cheap hotel"`, its dialog act is `{"Hotel-Inform":[["Price", "cheap"]]}`, and translated tag sequence is `["O", "O", "O", "B-Hotel-Inform+Price", "O"]`. A linear layer takes bert word embeddings as input and classify the tag label.
- For each of the other dialog acts, such as `(Hotel-Request, Address, ?)`, another linear layer takes embeddings of `[CLS]` of current utterance as input and do the binary classification. If you set `context=true` in config file, utterances of last three turn will be concatenated and provide context information with embedding of `[CLS]` for classification.  

We fine-tune BERT parameters on multiwoz.

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
$ python train.py --config_path multiwoz/configs/[config].json
```

The model will be saved on `output/[config]/bestcheckpoint.tar`. Also, it will be zipped as `output/[config]/bert_multiwoz_[config].zip`. 

#### Predict

See `nlu.py` for usage:

## Data

We use the multiwoz data (`data/multiwoz/[train|val|test].json.zip`).

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

