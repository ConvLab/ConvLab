# Adding non-RL policy to the pipeline

This is a tutorial for adding non-RL external policy to the ConvLab environment.

## Adding new model

Add all of your file related to non-RL policy under  ```./convlab/modules/policy/system``` or under ```./convlab/modules/policy/system/[Domain]```.




 ```convlab/agent/algorithm``` is the directory for the RL policies. However, when one is interested in plugging in one's own non-RL model to the ConvLab framework, one can do so by adding the new model under  ```./convlab/modules/policy/system``` or under ```./convlab/modules/policy/system/[Domain]``` e.g. ./convlab/modules/policy/system/multiWoz.



## Config set up
