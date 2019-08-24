# ConvLab
ConvLab is an open-source multi-domain end-to-end dialog system platform, aiming to enable researchers to quickly set up experiments with reusable components and compare a large set of different approaches, ranging from conventional pipeline systems to end-to-end neural models, in common environments.

## Package Overview
<table>
<tr>
    <td><b> convlab </b></td>
    <td> an open-source multi-domain end-to-end dialog research library </td>
</tr>
<tr>
    <td><b> convlab.agent </b></td>
    <td> a module for constructing dialog agents including RL algorithms </td>
</tr>
<tr>
    <td><b> convlab.env </b></td>
    <td> a collection of environments </td>
</tr>
<tr>
    <td><b> convlab.experiment </b></td>
    <td> a module for running experiments at various levels </td>
</tr>
<tr>
    <td><b> convlab.evaluator </b></td>
    <td> a module for evaluating a dialog session with various metrics </td>
</tr>
<tr>
    <td><b> convlab.modules </b></td>
    <td> a collection of state-of-the-art dialog system component models including NLU, DST, Policy, NLG </td>
</tr>
<tr>
    <td><b> convlab.human_eval </b></td>
    <td> a server for conducting human evaluation using Amazon Mechanical Turk </td>
</tr>
<tr>
    <td><b> convlab.lib </b></td>
    <td> a library of common utilities </td>
</tr>
<tr>
    <td><b> convlab.spec </b></td>
    <td> a collection of experiment spec files </td>
</tr>
</table>

## Installation
ConvLab requires Python 3.6.5 or later. Windows is currently not offically supported.

### Installing via pip

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used to set up a virtual environment with the
version of Python required for ConvLab.  If you already have a Python 3.6 or 3.7
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6.5

    ```bash
    conda create -n convlab python=3.6.5
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use ConvLab.

    ```bash
    source activate convlab
    ```

#### Installing the library and dependencies

Installing the library and dependencies is simple using `pip`.

   ```bash
   pip install -r requirements.txt
   ```
If your Linux system does not have essential building tools installed, you might need to install it by running
 ```bash
 sudo apt-get install build-essential
 ```
ConvLab uses 'stopwords' in nltk, and you need to download it by running
```bash
python -m nltk.downloader stopwords
```

#### Installation tips on CentOS
Please refer to the instructions here: https://github.com/daveta/convlab-notes/wiki/Installing-Convlab-on-Centos7 

### Installing using Docker

Docker provides more isolation and consistency, and also makes it easy to distribute your environment to a compute cluster.

Once you have [installed Docker](https://docs.docker.com/engine/installation/) just run the following commands to get an environment that will run on either the CPU or GPU.

1. Pull docker </br>
```docker pull convlab/convlab:0.2.2```

2. Run docker </br>
```docker run -it --rm convlab/convlab:0.2.2```

## Running ConvLab
Once you've downloaded ConvLab and installed required packages, you can run the command-line interface with the `python run.py` command.
```bash
$ python run.py {spec file} {spec name} {mode}
```

For non-RL policies:
```bash
# to evaluate a dialog system consisting of NLU(OneNet), DST(Rule), Policy(Rule), NLG(Template) on the MultiWOZ environment
$ python run.py demo.json onenet_rule_rule_template eval

# to see natural language utterances 
$ LOG_LEVEL=NL python run.py demo.json onenet_rule_rule_template eval

# to see natural language utterances and dialog acts 
$ LOG_LEVEL=ACT python run.py demo.json onenet_rule_rule_template eval

# to see natural language utterances, dialog acts and state representation
$ LOG_LEVEL=STATE python run.py demo.json onenet_rule_rule_template eval
```

For RL policies:
```bash
# to train a DQN policy with NLU(OneNet), DST(Rule), NLG(Template) on the MultiWOZ environment
$ python run.py demo.json onenet_rule_dqn_template train

# to use the policy trained above (this will load up the onenet_rule_dqn_template_t0_s0_*.pt files under the output/onenet_rule_dqn_template_{timestamp}/model directory)
$ python run.py demo.json onenet_rule_dqn_template eval@output/onenet_rule_dqn_template_{timestamp}/model/onenet_rule_dqn_template_t0_s0
```

Note that currently ConvLab can only train the policy component by interacting with a user simulator. 
For other components, ConvLab supports offline supervise learning. For example, you can train an NLU model using the local training script as in [OneNet](https://github.com/ConvLab/ConvLab/tree/dev/convlab/modules/nlu/multiwoz/onenet).

## Creating a new spec file
A spec file is used to fully specify experiments including a dialog agent and a user simulator. It is a JSON of multiple experiment specs, each containing the keys agent, env, body, meta, search.

We based our implementation on [SLM-Lab](https://github.com/kengz/SLM-Lab/tree/master/slm_lab). For an introduction to these concepts, you should check [these docs](https://kengz.gitbooks.io/slm-lab/content/).

Instead of writing one from scratch, you are welcome to modify the `convlab/spec/demo.json` file. Once you have created a new spec file, place it under `convlab/spec` directory and run your experiments. Note that you don't have to prepend `convlab/spec/` before your spec file name.

## Participation in DSTC-8
1. Extend ConvLab with your code, and include submission.json under the convlab/spec directory.
2. In submission.json, specify up to 5 specs with the name submission[1-5].
2. Make sure the code with the config is runnable in the docker environment.
3. If your code uses external packages beyond the existing docker environment, please choose one of the following two approaches to specify your environment requirements:
    - Add install.sh under the convlab directory. install.sh should include all required extra packages.
    - Create your own Dockerfile with the name dev.dockerfile
4. Zip the system and submit.
### Evaluation
1. Automatic end2end Evaluation: The submitted system will be evaluated using the user-simulator setting in spec `milu_rule_rule_template` in `convlab/spec/baseline.json`. We will use the evaluator MultiWozEvaluator in `convlab/evaluator/multiwoz` to report metrics including success rate, average reward, number of turms, precision, recall, and F1 score.
2. Human Evaluation: The submitted system will be evaluated in Amazon Mechanic Turk. Crowd-workers will communicate with your summited system, and provide a rating based on the whole experience (language understanding, appropriateness, etc.)
## Contributions
The ConvLab team welcomes contributions from the community. Pull requests must have one approving review and no requested changes before they are merged. The ConvLab team reserves the right to reject or revert contributions that we don't think are good additions.

## Citing
If you use ConvLab in your research, please cite [ConvLab: Multi-Domain End-to-End Dialog System Platform](https://arxiv.org/abs/1904.08637).
```
@inproceedings{lee2019convlab,
  title={ConvLab: Multi-Domain End-to-End Dialog System Platform},
  author={Lee, Sungjin and Zhu, Qi and Takanobu, Ryuichi and Li, Xiang and Zhang, Yaoqin and Zhang, Zheng and Li, Jinchao and Peng, Baolin and Li, Xiujun and Huang, Minlie and Gao, Jianfeng},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year={2019}
}
```
