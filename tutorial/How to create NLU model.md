# Adding NLU model to the pipeline

This is a tutorial for adding an NLU model to the ConvLab environment. To demonstrate, we will walk through an example of incorporating "SVMNLU" on the "multiwoz" domain.



## Adding a new model

Add all of your NLU model factors under your model folder: ```convlab/modules/nlu/[domain]/[your_model_name]``` which we will refer as ```$model_folder``` for simplicity. We have the example codes under ```convlab/modules/nlu/multiwoz/svm``` for the case of using "multiwoz" as domain and "SVMNLU" as your model name.


In order to incorporate your NLU model into the ConvLab environment, first create your NLU model class in ```$model_folder/nlu.py``` , inheriting the provided interface: ```convlab/modules/nlu/nlu.py```. 
The ```nlu.py``` should have the following functionalities:

1. Load the model in ```__init__```. You can 1) load from local path or 2) load from url using `cached_path` from `convlab.lib.file_util`
2. Provide custom ```parse()``` function to obtain an output dialog act from your model using current user utterance.



## Making the new model visible

In order to make the new policy algorithm visible to the ConvLab environment, one has to include new class name ```__init__.py``` on each level accordingly.

For example, to make the class ```SVMNLU``` class inside ```convlab/modules/nlu/multiwoz/svm/nlu.py``` visible, we added

- ```from convlab.modules.nlu.multiwoz.svm.nlu import SVMNLU``` in
  - ```convlab/modules/nlu/multiwoz/__init__.py```.

* ```from convlab.modules.nlu.multiwoz import SVMNLU``` in
  * ```convlab/modules/nlu/__init__.py ```



## Config set up

Then just config the class name and parameters of your model in the configuration file, you can use your NLU for end2end evaluation. An example configuration is `svmnlu_rule_rule_template` in ```convlab/spec/demo.json```. The `name` must be your model class name in ```$model_folder/nlu.py```.

If you want to evaluate the model in multiwoz dataset, you can refer to ```convlab/modules/nlu/multiwoz/evaluate.py```.



