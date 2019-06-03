Confusion Network Config Options                 {#CNetConfig}
================================

## [grammar]
* `acts` - a JSON list of all the possible act-types
* `nonempty_acts` - a JSON list of those acts that require a slot-value pair
* `ontology` - a JSON file with the ontology for the domain, this is found in a belief tracking corpus
* `slots_enumerated` - a JSON list of slots for which we do not tag values with <generic_value>


## [classifier]
* `type` - the type of classifier used. {*svm*}
* `features` - a JSON list of the features extracted from a turn. {"cnet", "valueIdentifying","nbest","lastSys","nbestLengths","nbestScores"}. These refer to classes in `Features.py`.
* `max_ngram_length` - the maximum length of ngrams to extract. Default is 3.
* `max_ngrams` - the maximum number of ngrams to extract per turn. Default is 200.
* `skip_ngrams` - Whether to use skip ngrams, for nbest features {"True","False"}
* `skip_ngram_decay` - Factor to discount skips by. Default is 0.9.
* `min_examples` - the minimum number of positive examples of a tuple we require to train a classifier for it. Default is 10.


## [train]
* `output` - the pickle file where the learnt classifier is saved
* `dataset` - a JSON list of dataset names to use for training
* `dataroot` - the directory where the data is found
* `log_input_key` - the key to use of `log_turn["input"]`. Default is 'batch'.

## [decode]
* `output` - the JSON file that decoder output is saved to. Compatible with belief tracker scoring scripts
* `dataset` - a JSON list of dataset names to decode
* `dataroot` - the directory where the data is found
* `max_active_tuples` - the maximum number of tuples to consider in a decode. Default is 10.
* `tail_cutoff` - the minimum weight of an SLU hypothesis to include. Default 0.001
* `log_input_key` - the key to use of `log_turn["input"]`. Default is 'batch'.

## [evaluate]
* `csv_output` - the CSV file where the results of `score_slu.py` are saved.
* `report_output` - text file with the report from `report_slu.py`.
* `tracker_output` - JSON file where a baseline tracker output is saved, run on the output of SLU.

## [export]
* `models` - the text file to contain all the SVMs, one after the other. An SVM is saved in libsvm's sparse export format, with the tuple at the beginning, and terminated with a period then a new line.
* `dictionary` - JSON file with the mapping from features to vector indices. This is a JSON list of the feature keys, sorted by their mapping index.
* `config` - file to save any options in Caesar's config format that need to be set to run the decoder. E.g. max ngram length

## [experiment]
* `type` - the experiment type {vary_train}
* `repeat` - number of times to repeat each configuration. Default is 1.
* `vary` - What parameters in the config file should be changed. E.g:

    [
        ["section_name1","option_name1", ["value11", "value12",...]],
        ["section_name2","option_name2", ["value21", "value22",...]]
    ]
* `num_processes` - How many concurrent processes to use. Default is 1.



# Confusion Network Tutorial                             {#CNetTrainTutorial}

A list of configuration options are provided in \subpage CNetConfig.

# Training, testing, and saving a CNet decoder

This tutorial will explain how to create a CNet decoder that can be used by Caesar. It also explains how to run an experiment where you vary training parameters, and see what worked better. For full documentation of the config variables, please see `config.md`.

## Install sklearn (lmr46: installing the last sklearn version with pip should work)

pip install -U scikit-learn

## OLD instruction ...

[Download a copy](http://scikit-learn.org/stable/install.html) of `scikit-learn-0.14.tar.gz`, untar it, then run `python setup.py install --user`.

## Get a copy of a belief tracker corpus

E.g. the DSTC II data, which can be found here: https://bitbucket.org/matthen/dstc-ii .
Get a `*.tar.gz` and extract it into `./data` :

```
mkdir corpora
cd corpora
wget https://bitbucket.org/matthen/dstc-ii/downloads/DSTCCAM.tar.gz
tar -xzf DSTCCAM.tar.gz
```

This should provide `corpora/data` and `corpora/scripts`. The `scripts` should contain Python classes for looping through the data (`dataset_walker.py`), and scoring the output of a semantic decoder (`score_slu.py`).

## Train model

We will start building up a config file, which defines the model. Model settings are defined in the `[classifier]` section, and then following sections define settings for python scripts of that name. For example here, we add a section called `[train]` which is used by `train.py`. 

Create `config/eg.cfg` with the following contents:

```
[DEFAULT]
output_dir = output
name = eg

[grammar]
acts = ["inform","request","deny","negate","confirm","null","repeat","affirm","bye","reqalts","hello","thankyou","ack","help"]
nonempty_acts = ["inform","confirm","request","deny"]

slots_enumerated = ["area","pricerange"]
ontology = corpora/scripts/config/ontology_Oct11.json
```

​    
​    [classifier]
​    type = svm
​    features = ["cnet"]

​    
​    [train]
​    output = %(output_dir)s/%(name)s.pickle
​    dataset = ["Oct11_train"]
​    dataroot = corpora/data

The `[grammar]` section contains the grammar of user acts, which the decoder needs to know. Running `python checkGrammar config/eg.cfg` will check that there are no contradictory user acts in the train and test sets. For this example, `checkGrammar.py` should output:

```
Checking  train
undeclared informable slots found 
[u'task', u'type']
```

we don't want to model `task` and `type`, because they are always `find` and `restaurant` respectively. Omitting them from the grammar means they will just be ignored.

In the `[classifier]` section, we define the features used as a JSON list of strings. These correspond to classes in `Features.py`.

Do:

```
mkdir output
python train.py config/eg.cfg
```

This will take a while to run, and in the end should create `output/eg.pickle` with the trained model. At the end of training, it will tell you something like:

```
Not able to learn about: 
(u'confirm', u'name', (generic value for name (None))), (u'confirm', u'pricerange', u'moderate'), (u'restart',), 
(u'deny', u'pricerange', 'dontcare'), (u'confirm', u'pricerange', 'dontcare'), (u'confirm', u'area', 'dontcare'), 
(u'deny', u'area', 'dontcare'), (u'deny', u'name', (generic value for name (None))), (u'deny', u'area', u'north'), 
(u'deny', u'area', u'south'), (u'deny', u'pricerange', u'moderate'), (u'confirm', u'pricerange', u'expensive'), 
(u'confirm', u'pricerange', u'cheap')
```

These are the tuples that weren't represented in the training data, and so couldn't be learnt.

## Decode a test set

We will use the decoder to decode a test set. Add a `[decode]` section to the config file:

```
[decode]
output = %(output_dir)s/%(name)s.decode.json
; this will be the output of the decoder on the test set
dataset = ["Oct11_test"]
dataroot = corpora/data
```

Then run:

```
python decode.py config/eg.cfg
```

Now there should be a file `output/eg.decode.json` with the decoder's output.

## Evaluate the decoding results

The `score_slu.py` and `report_slu.py` scripts packaged with the belief tracking corpus can be run on the output of `decode.py`, if you use the correct command-line arguments. `evaluate.py` takes the config file as its single argument, runs the SLU scoring and creates a report. 

Add the following section to the config file:

```
[evaluate]
csv_output = %(output_dir)s/%(name)s.score.csv
report_output = %(output_dir)s/%(name)s.report.txt
```

Run `python evaluate.py config/eg.cfg`. This will create the `output/eg.score.csv` file and `output/eg.report.txt` The metrics can be found in `output/eg.score.csv`.
    

## Run an experiment

Copy the config file: `cp config/eg.cfg config/eg_experiment.cfg`. Delete the `[DEFAULT]` section and add:

```
[experiment]
name: feature_set 
type: vary_train   ; this type of experiment will train a bunch of models and track the track dataset
vary:   [
            ["classifier", "features", [
                "[\"cnet\"]",
                "[\"nbest\"]"
                ]]
        ]
; section, option, values
```

Be sure to `mkdir output/experiments`. This experiment will try each possible feature set and output the results to `output/experiments/feature_set`. Here we are comparing ngram counts derived from the confusion network, versus using the nbest list. You can also vary the values of more options like this:

```
[
    ["section_name1","option_name1", ["value11", "value12",..]],
    ["section_name2","option_name2", ["value21", "value22",..]]
]
```

The for `vary_train` experiments, the `experiment.py` script will try all possible combinations of these options, and run train, decode, evaluate. It sets the `DEFAULT, name` option automatically for each run, so if you have used `%(name)s` throughout your config as above, it will work fine. The script uses `multiprocessing` to use multiple processes to evaluate runs. Set the `num_processes` option to configure this (default is 1).

Now run `python experiment.py config/eg_experiment.cfg`.

This will start by printing:

```
Configuring:
        run_0
         Setting:
                classifier_features = ["cnet"]
putting  run_0
Configuring:
        run_1
         Setting:
                classifier_features = ["nbest"]
putting  run_1
```

Try listing the contents of the experiment directory:

```
ls output/experiments/feature_set/
experiment_config.cfg	log.txt			run_0.cfg		run_1.cfg
```

Note that a copy of the config you used to create the experiment is created. This is useful for recreating results. `run_0.cfg` and `run_1.cfg` will also allow you to recreate individual configurations. When the experiment finishes there will be more files in the directory, including the `scores.csv` files:

```
$head -20 output/experiments/feature_set/run_*.score.csv
==> output/experiments/feature_set/run_0.score.csv <==
belief_accuracy,all_acc,           0.96242
belief_accuracy,all_l2,            0.04696
belief_accuracy,all_logp,         -0.15237
(ommitted goal, requested, and method breakdown )
ice,ICE,                           1.02352
tophyp,fscore,                     0.87771
tophyp,precision,                  0.90081
tophyp,recall,                     0.85577

==> output/experiments/feature_set/run_1.score.csv <==
belief_accuracy,all_acc,           0.96221
belief_accuracy,all_l2,            0.05005
belief_accuracy,all_logp,         -0.16502
(ommitted goal, requested, and method breakdown )
ice,ICE,                           1.11565
tophyp,fscore,                     0.86706
tophyp,precision,                  0.89549
tophyp,recall,                     0.84038
```

​    
(Confusion network features are `run_0`, and perform slightly better on all metrics.)

Running `python experiment.py config/eg.cfg` (or with any config without an `[experiment]` section) is equivalent to `python train.py config/eg.cfg; python decode.py config/eg.cfg; python evaluate.py config/eg.cfg`.

## Output for Caesar

Add the following section to the config file:

```
[export]
models = %(output_dir)s/%(name)s.caesar.svms.txt
dictionary = %(output_dir)s/%(name)s.caesar.dic.txt
config  = %(output_dir)s/%(name)s.caesar.cfg
```

Caesar needs two files:

- `models` contains all the SVMs, one after the other. An SVM is saved in libsvm's sparse export format, with the tuple at the beginning, and terminated with a period then a new line.
- `dictionary` is the mapping from features to vector indices. This is a JSON list of the feature keys, sorted by their mapping index.

The `config` contains any options in Caesar's config format that need to be set to run the decoder. The output for this example is:

```
# Automatically generated by CNetTrain scripts
        CNET   : MAX_NGRAMS           = 200
        CNET   : FEATURES             = ["cnet"]
        CNET   : DICTIONARY           = /Users/matt/Projects/vocaliq/SemIO/CNetTrain/output/eg.caesar.dic.txt
        CNET   : MAX_NGRAM_LENGTH     = 3
        CNET   : MODELS               = /Users/matt/Projects/vocaliq/SemIO/CNetTrain/output/eg.caesar.svms.txt
        CNET   : TAIL_CUTOFF          = 0.001
        CNET   : MAX_ACTIVE_TUPLES    = 10
```

Include this config in your master config, and the decoder should work. Check the absolute paths are okay, and then consider doing a `#include`.

## Other config variables

Every config variable is documented in `config.md` in the root of the `CNetTrain` directory. 