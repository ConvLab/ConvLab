# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from convlab.modules.usr import UserSimulator
from convlab.modules.nlu.multiwoz import ErrorNLU, OneNetLU, MILU, SVMNLU

from convlab.modules.dst.multiwoz import RuleDST
from convlab.modules.word_dst.multiwoz import MDBTTracker
from convlab.modules.policy.system.multiwoz import RuleBasedMultiwozBot, RuleInformBot, VanillaMLEPolicy
from convlab.modules.policy.user.multiwoz import UserPolicyAgendaMultiWoz
from convlab.modules.nlg.multiwoz import TemplateNLG, MultiwozTemplateNLG, SCLSTM
from convlab.modules.word_policy.multiwoz import MDRGWordPolicy

from convlab.modules.util import Log

import os

os.environ['PY_ENV'] = os.environ.get('PY_ENV') or 'development'
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

EVAL_MODES = ('eval')
TRAIN_MODES = ('search', 'train', 'dev')
