# Dialog Policy

In the pipeline task-oriented dialog framework, the dialog policy module
takes as input the dialog state, and chooses the system action bases on
it.

This directory contains the interface definition of dialog policy
module for both system side and user simulator side, as well as some
implementations.

## Interface

### system

The interfaces for system policy are defined in system.policy.SysPolicy:

- **predict** takes as input agent state (often the state tracked by DST)
and outputs the next system action.

- **init_session** reset the model variables for a new dialog session.

### user simulator

The interfaces for user side are defined in user policy is defined in
user.policy.UserPolicy:

- **predict** takes as input dialog state and last system action, and
outputs the next user action. Note that the difference of user and system
predict interface is that the user predict interfaces takes system action
as input in addition.

- **init_session** reset the model variables for a new session.