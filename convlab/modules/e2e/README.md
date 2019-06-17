# End2end Model

End-to-end dialog models, which directly take user utterances as input
and gives a utterance as system response.
We implemented two end2end dialog models, including Sequicity and Mem2seq.

## Interface

The interfaces for the two end2end models are the same, which are defined in
multiwoz.Sequicity.Sequicity.Sequicity and multiwoz.Mem2Seq.Mem2Seq.Mem2Seq
respectively:

- **predict** takes as input the user utterance and returns a utterances as
user response.

- **reset** reset the internal variables for a new session.