# Adding non-RL policy to the pipeline

This is a tutorial for adding non-RL external policy to the ConvLab environment. To demonstrate, we will walk through an example of incorporating simple supervised model "Vanila MLE" on the "multiwoz" domain.

## Adding new model

Add all of your non-RL policy model factors under your model folder:  ```./convlab/modules/policy/system/[your_model_name]``` or ```./convlab/modules/policy/system/[domain]/[your_model_name]```. Using "multiwoz" as domain and "VanilaMLEPolicy" as your_model_name, we have an example codes under ```convlab/modules/policy/system/multiwoz/vanila_mle``` which we will call as ```$model_folder``` for a convenience.

In order to incorporate your non-RL policy model into the ConvLab environment, first create ```$model_folder/policy.py``` by inheriting the provided interface: ```/convlab/modules/policy/system/policy.py```. 

The ```policy.py``` should have following functionalities:

1. Provide the model definition and loading function. In the example, model definition and loading function, respectively, where the ```load_archive``` function is provided from allennlp.
```bash
$model_folder/model.py 
load_archive(archive_file, cuda_device=cuda_device)
``` 
<!-- In this tutorial, the link between model definition and loader is not visible due to allennlp implementation-->

2. The link that can enable access your defined dataset_reader, action_vocab (your defined output space), and state_encoder (input to your policy). In the example, 
```bash
from allennlp.data import DatasetReader
...
        self.dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.action_vocab = self.dataset_reader.action_vocab 
        self.state_encoder = self.dataset_reader.state_encoder
``` 

3. Provide ```predict()``` function to obtain an output action from your model.


```
```


 ```convlab/agent/algorithm``` is the directory for the RL policies. However, when one is interested in plugging in one's own non-RL model to the ConvLab framework, one can do so by adding the new model under  ```./convlab/modules/policy/system``` or under ```./convlab/modules/policy/system/[Domain]``` e.g. ./convlab/modules/policy/system/multiWoz.



## Config set up
