# Action Decoder

If you use prediction-based policy (e.g., DQN), the original output of policy
module is vector. Then you have to convert this vector to
structured dict variable using action decoder.


## Interface

The interfaces are defined in multiwoz.multiwoz_vocab_action_decoder.MultiWozVocabActionDecoder

- **decode** takes as input the vector action and dialog state, and convert the action
into a dict variable.