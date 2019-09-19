# Transformer

Transformer encodes the user utterance and decodes the system utterance with a stacked Transformer architecture where self-attention and multi-head attention mechanism is used. It's first proposed at [NIPS 2017](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) . Here we appends the dialog act vector to the input word embedding to feed the semantic information into Transformer.

# Run the code

TRAIN

```sh
$ PYTHONPATH=../../../.. python train.py
```

TEST

```sh
$ PYTHONPATH=../../../.. python train.py --option test
```

# Data

We use the multiwoz data under the `data` directory, the trained model is saved at `checkpoints` directory .

# Reference

```
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}
```