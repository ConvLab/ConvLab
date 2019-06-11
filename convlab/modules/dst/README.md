# Dialog State Tracking

In the pipeline task-oriented dialog framework, the DST module encodes
the dialog history into a pre-defined state representation, which can be
either structured or unstructured.
In each turn, it takes as input the dialog act of user utterances, and updates
its internal state variable.

This directory contains the interface definition of dialog state
tracking module and some act-level DST module implementations.

## Interface

The interfaces of DST are defined in state_tracker.Tracker, including:

- **update** takes as input the new observation in each turn, and update
the internal state variable of DST component. The new observation is the
dialog act of user utterance, which may be the output of NLU module.

- **reset**  reset the internal state variable for a new dialog session.