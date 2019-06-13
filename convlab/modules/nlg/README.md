# Natural Language Generation

In a pipeline task-oriented dialog framework, the NLG module takes as input
the dialog act of system action, and convert it to a natural language
utterance, which is the system response that user receives.

This directory contains the interface definition of natural language
generation module.


## Interface

The interfaces are defined in nlg.NLG:

- **generate** takes as input dialog act, and converts it into a natural
language utterance.