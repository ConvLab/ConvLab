# Natural Language Understanding

In a pipeline task-oriented dialog framework, the NLU module takes as input raw
user utterance, and converts it into a dialog act format.

This directory contains the interface definition of natural language understanding module and some implementations.

## Interface

The interface of NLU is defined in nlu.NLU, including:

- **parse** takes as input a natural langauge utterance, and outputs its dialog act.