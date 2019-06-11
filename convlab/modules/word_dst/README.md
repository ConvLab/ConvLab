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