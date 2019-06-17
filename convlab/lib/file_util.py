# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

from allennlp.common.file_utils import cached_path as allennlp_cached_path


def cached_path(file_path, cached_dir=None):
    if not cached_dir:
        cached_dir = str(Path(Path.home() / '.convlab') / "cache")

    return allennlp_cached_path(file_path, cached_dir)
