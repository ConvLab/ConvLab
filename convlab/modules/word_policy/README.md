# Word-level Dialog Policy

In the pipeline task-oriented dialog framework, the dialog policy module
takes as input the dialog state, and chooses the system action bases on
it. The word-level policy module directly reads from raw input utterances,
and chooses the next system action.

## Interface

Word-level policy shares the same interface with system policy
(convlab.modules.policy.system.policy.SysPolicy). The difference is in the
internal implementation of *predict* method, in which the word-level model
directly reads from natural language utterances.

- **predict** takes as input agent state (often the state tracked by DST)
and outputs the next system action.

- **init_session** reset the model variables for a new dialog session.