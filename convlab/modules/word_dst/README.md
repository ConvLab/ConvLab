# Word-level Dialog State Tracking

In the pipeline task-oriented dialog framework, the DST module encodes
the dialog history into a pre-defined state representation, which can be
either structured or unstructured.
The word-level DST directly takes as input the user utterances and updates
the dialog state.

This directory contains the interface definition of word-level dialog state
tracking module and some implementations.

## Interface

Word-level DST shares the same interfaces with DA-level DST
(defined in convlab.modules.dst.state_tracker.Tracker). The difference is
that the input of *update* method is the raw user utterance instead of
is dialog act.

- **update** takes as input the new observation in each turn, and update
the internal state variable of DST component. The new observation is the
dialog act of user utterance, which may be the output of NLU module.

- **reset**  reset the internal state variable for a new dialog session.

## Evaluation

If you want to evaluate the word DST model, you can run the evaluate.py
script under the model directory:

````bash
word_dst/multiwoz$ python evaluate.py
````

It will record the test results in the `word_dst_test_result.json` file.
The accuracy numbers, including domain-level, slot-level and session-level
will be shown in the terminal.