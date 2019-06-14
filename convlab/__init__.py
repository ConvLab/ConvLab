# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

os.environ['PY_ENV'] = os.environ.get('PY_ENV') or 'development'
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

EVAL_MODES = ('eval')
TRAIN_MODES = ('train', 'dev')
