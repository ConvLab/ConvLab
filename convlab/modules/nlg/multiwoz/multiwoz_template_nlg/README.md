# Multiwoz Template NLG

Template NLG for Multiwoz dataset. The templates are extracted from data and modified manually.



## Quick start

There are three mode:

- `auto`: templates extracted from data without manual modification, may have no match (return 'None');
- `manual`: templates with manual modification, sometimes verbose;
- `auto_manual`: use auto templates first. When fails, use manual templates.

Example:

```python
from convlab.modules.nlg.multiwoz.multiwoz_template_nlg.multiwoz_template_nlg import MultiwozTemplateNLG

# dialog act
dialog_acts = {'Train-Inform': [['Day', 'wednesday'], ['Leave', '10:15']]}
# whether from user or system
is_user = False

multiwoz_template_nlg = MultiwozTemplateNLG()
print(dialog_acts)
print(multiwoz_template_nlg.generate(dialog_acts, is_user, mode='manual'))
print(multiwoz_template_nlg.generate(dialog_acts, is_user, mode='auto'))
print(multiwoz_template_nlg.generate(dialog_acts, is_user, mode='auto_manual'))
```
Result:
```
{'Train-Inform': [['Day', 'wednesday'], ['Leave', '10:15']]}
The train is for wednesday you are all set. I have a train leaving at 10:15 would that be okay ?
I can help you with that . one leaves wednesday at 10:15 , is that time okay for you ?
There is a train leaving at 10:15 on wednesday .
```



## Templates

This directory contains all extracted templates (*.json). Generally, we select the utterances that have only one dialog act to extract templates. For `auto` mode, the templates may have several slot, while for `manual` mode, the templates only have one slot. As a result, `auto` templates can fail when some slot combination don't appear in dataset, while for `manual` mode, we generate utterance slot by slot, which could not fail but may be verbose. Notice that `auto` templates could be inappropriate.



## Generation

For most dialog act, we fill the slots in template with corresponding values. For `general` and `Request` dialog act, there are no slot in templates. For `Select` dialog act in `manual` mode, we write a simple rule.
