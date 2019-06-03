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
    <td><b> convlab.modules </b></td>
    <td> a collection of state-of-the-art dialog system component models including NLU, DST, Policy, NLG </td>
</tr>
<tr>
    <td><b> convlab.human_eval </b></td>
    <td> a server for conducting human evaluation using Amazon Mechanical Turk </td>
</tr>
<tr>
    <td><b> convlab.lib </b></td>
    <td> a libarary of common utilities </td>
</tr>
<tr>
    <td><b> convlab.spec </b></td>
    <td> a collection of experiment spec files </td>
</tr>
</table>

## Running ConvLab
Once you've downloaded ConvLab and installed required packages, you can run the command-line interface with the `python run.py` command.

```bash
$ python run.py {spec file} {spec name} {mode}
```

For example:
```bash
# to evaluate a dialog system consisting of NLU(OneNet), DST(Rule), Policy(Rule), NLG(Template) on the MultiWOZ environment
$ python run.py demo.json onenet_rule_rule_template eval

# to see natural language utterances 
$ LOG_LEVEL=NL python run.py demo.json onenet_rule_rule_template eval

# to see natural language utterances and dialog acts 
$ LOG_LEVEL=ACT python run.py demo.json onenet_rule_rule_template eval

# to see natural language utterances, dialog acts and state representation
$ LOG_LEVEL=STATE python run.py demo.json onenet_rule_rule_template eval

# to train a DQN policy with NLU(OneNet), DST(Rule), NLG(Template) on the MultiWOZ environment
$ python run.py demo.json onenet_rule_dqn_template train

# to use the policy trained above
$ python run.py output/onenet_rule_dqn_template_{timestamp}/onenet_rule_dqn_template_spec.json onenet_rule_dqn_template eval@onenet_rule_dqn_template_t0_s0
```

Note that currently ConvLab can only train the policy component by interacting with a user simulator. 
For other components, ConvLab supports offline supervise learning. For example, you can train a NLU model using the local training script as in [OneNet](https://github.com/ConvLab/ConvLab/tree/dev/convlab/modules/nlu/multiwoz/onenet).

## Creating a new spec file
A spec file is used to fully specify experiments including a dialog agent and a user simulator. It is a JSON of multiple experiment specs, each containing the keys agent, env, body, meta, search.

We based our implementation on [SLM-Lab](https://github.com/kengz/SLM-Lab/tree/master/slm_lab). For an introduction to these concepts, you should check [these docs](https://kengz.gitbooks.io/slm-lab/content/).

Instead of writing one from scratch, you are welcome to modify the `convlab/spec/demo.json` file. Once you have created a new spec file, place it under `convlab/spec` directory and run your experiments. Note that you don't have to prepend `convlab/spec/` before your spec file name.

## Contributions
The ConvLab team welcomes contributions from the community. Pull requests must have one approving review and no requested changes before they are merged. The ConvLab team reserve the right to reject or revert contributions that we don't think are good additions.

## Citing
If you use ConvLab in your research, please cite [ConvLab: Multi-Domain End-to-End Dialog System Platform](https://arxiv.org/abs/1904.08637).
```
@inproceedings{lee2019convlab,
  title={ConvLab: Multi-Domain End-to-End Dialog System Platform},
  author={Lee, Sungjin and Zhu, Qi and Takanobu, Ryuichi and Li, Xiang and Zhang, Yaoqin and Zhang, Zheng and Li, Jinchao and Peng, Baolin and Li, Xiujun and Huang, Minlie and others},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year={2019}
}
```
