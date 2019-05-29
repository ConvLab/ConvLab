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