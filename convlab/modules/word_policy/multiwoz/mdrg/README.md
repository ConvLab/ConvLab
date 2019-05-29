# multiwoz

multiwoz is an open source toolkit for building end-to-end trainable task-oriented dialogue models.
It is released by Paweł Budzianowski from Cambridge Dialogue Systems Group under Apache License 2.0.

# Requirements
Python 2 with pip

# Quick start
In repo directory:

## Preprocessing
To download and pre-process the data run:

```python create_delex_data.py```

## Training
To train the model run:

```python train.py [--args=value]```
```
python train.py --max_epochs 20 --batch_size 64 --lr_rate 0.005 --clip 5.0 --l2_norm 0.00001 --dropout 0.0 --optim Adam --emb_size 50 --use_attn True --hid_size_enc 150 --hid_size_pol 150 --hid_size_dec 150 --cell_type lstm --cuda 0
```


Some of these args include:

```
// hyperparamters for model learning
--max_epochs        : numbers of epochs
--batch_size        : numbers of turns per batch
--lr_rate           : initial learning rate
--clip              : size of clipping
--l2_norm           : l2-regularization weight
--dropout           : dropout rate
--optim             : optimization method

// network structure
--emb_size          : word vectors emedding size
--use_attn          : whether to use attention
--hid_size_enc      : size of RNN hidden cell
--hid_size_pol      : size of policy hidden output
--hid_size_dec      : size of RNN hidden cell
--cell_type         : specify RNN type
```

## Testing
To evaluate the run:

```python test.py [--args=value]```

# Benchmark results
The following [benchmark results](http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/) were produced by this software.
We ran a small grid search over various hyperparameter settings
and reported the performance of the best model on the test set.
The selection criterion was 0.5*match + 0.5*success+100*BLEU on the validation set.
The final parameters were:

```
// hyperparamters for model learning
--max_epochs        : 20
--batch_size        : 64
--lr_rate           : 0.005
--clip              : 5.0
--l2_norm           : 0.00001
--dropout           : 0.0
--optim             : Adam

// network structure
--emb_size          : 50
--use_attn          : True
--hid_size_enc      : 150
--hid_size_pol      : 150
--hid_size_dec      : 150
--cell_type         : lstm
```


# References
If you use any source codes or datasets included in this toolkit in your
work, please cite the corresponding papers. The bibtex are listed below:
```
[Budzianowski et al. 2018]
@inproceedings{budzianowski2018large,
    Author = {Budzianowski, Pawe{\l} and Wen, Tsung-Hsien and Tseng, Bo-Hsiang  and Casanueva, I{\~n}igo and Ultes Stefan and Ramadan Osman and Ga{\v{s}}i\'c, Milica},
    title={MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2018}
}

[Ramadan et al. 2018]
@inproceedings{ramadan2018large,
  title={Large-Scale Multi-Domain Belief Tracking with Knowledge Sharing},
  author={Ramadan, Osman and Budzianowski, Pawe{\l} and Gasic, Milica},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  volume={2},
  pages={432--437},
  year={2018}
}
```

# Bug Report

If you have found any bugs in the code, please contact: pfb30 at cam dot ac dot uk
