# Adding a non-RL policy to the pipeline

This is a tutorial for adding a non-RL external policy to the ConvLab environment. To demonstrate, we will walk through an example of incorporating a simple supervised model "VanilaMLEPolicy" on the "multiwoz" domain.

## Adding a new model

Add all of your non-RL policy model factors under your model folder: ```convlab/modules/policy/system/[domain]/[your_model_name]``` which we will refer as ```$model_folder``` for simplicity. We have the example codes under ```convlab/modules/policy/system/multiwoz/vanila_mle``` for the case of simple supervised model using "multiwoz" as domain and "VanilaMLEPolicy" as your_model_name.
<!-- ```./convlab/modules/policy/system/[your_model_name]``` or  -->

In order to incorporate your non-RL policy model into the ConvLab environment, first create ```$model_folder/policy.py``` by inheriting the provided interface: ```/convlab/modules/policy/system/policy.py```. 
The ```policy.py``` should have the following functionalities:

1. Provide the model loading function. In the example, the loading function, which is provided from allennlp, is ```load_archive(archive_file, cuda_device=cuda_device)```. c.f.) In the case of allennlp, the model definition (```$model_folder/model.py ```) is linked to the model loader implicitly by ```$model_folder/policy.py``` importing the model definition file. 
<!-- In this tutorial, the link between model definition and loader is not visible due to allennlp implementation-->

2. The link that can enable access your defined dataset_reader, action_vocab (your defined output space), and state_encoder (input to your policy). In the example, 
```bash
from allennlp.data import DatasetReader
...
        self.dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.action_vocab = self.dataset_reader.action_vocab 
        self.state_encoder = self.dataset_reader.state_encoder
``` 
3. Provide custom ```predict()``` function to obtain an output action from your model.

## Making the new model visible

In order to make the new policy algorithm visible to the ConvLab environment, one has to include new class name ```__init__.py``` on each level accordingly.

For example, to make the class ```VanilaMLEPolicy``` class inside ```convlab/modules/policy/system/multiwoz/vanila_mle/policy.py``` visible, we added
* ```from convlab.modules.policy.system.multiwoz import VanilaMLEPolicy``` importing line in
  * ```convlab/modules/policy/__init__.py``` 
* ```from convlab.modules.policy.system.multiwoz.vanilla_mle.policy import VanillaMLEPolicy``` importing line in
  * ```convlab/modules/policy/system/__init__.py``` 
  * ```convlab/modules/policy/system/multiwoz/__init__.py```.

## Config set up
To incorporate diverse policies, convlab has a placeholder called external policy (```convlab/agent/algorithm/external.py```), 
* Make your "algorithm" name in the config file (e.g. demo.json) as "ExternalPolicy" to use this funcationality. 

In the ```convlab/agent/algorithm/external.py```, the following three lines link the external policy algorithm you defined.
```bash
        params = deepcopy(ps.get(self.algorithm_spec, 'policy'))
        PolicyClass = getattr(policy, params.pop('name'))
        self.policy = PolicyClass(**params)
```
Unlike, other RL algorithms that are pre-defined, all algorithms that follow "ExternalPolicy" requires to have a key "policy" under "algorithm" and under this "policy" section, we get to specify the details of the new policy algorithm. To see more in detail, examine the two examples given below extracted from ```convlab/spec/demo.json```.

**Example 1 (Rule-based external policy)**
```bash
        "algorithm": {
                "name": "ExternalPolicy",
        "policy": {
                "name": "RuleBasedMultiwozBot"
        },
        "action_pdtype": "Argmax",
        "action_policy": "default"
```
**Example 2 (Vanila MLE external policy)**
```bash
        "algorithm": {
                "name": "ExternalPolicy",
        "policy": {
                "name": "VanillaMLEPolicy",
                "model_file": "https://convlab.blob.core.windows.net/models/vmle.tar.gz"
        },
        "action_pdtype": "Argmax",
        "action_policy": "default"
```


## Summary

 ```convlab/agent/algorithm``` is the directory for the RL policies. However, when one is interested in plugging in one's own non-RL model to the ConvLab framework, one can do so by adding the new model under  ```./convlab/modules/policy/system``` or under ```./convlab/modules/policy/system/[Domain]``` e.g. ./convlab/modules/policy/system/multiwoz. This model is linked through the whole pipeline through ExternalPolicy class at ```convlab/agent/algorithm/external.py```.



