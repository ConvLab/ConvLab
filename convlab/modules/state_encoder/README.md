# State Encoder

The state produced by DST module is often represented by a dict variable.
However, most existing policy models (e.g., DQN and Policy Gradient) takes as
input a vector. Therefore, you have to convert the state to a vector representation
before passing it to the policy module.

## Interface
The interfaces of state encoder for multiwoz domain task is defined in
multiwoz.multiwoz_state_encoder.MultiWozStateEncoder, including:

- **encode** takes as input the dict-format state, and returns the vector
representation of state.