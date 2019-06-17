# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import zipfile
from typing import Dict, List, Any

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, LabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from convlab.lib.file_util import cached_path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("onenet")
class OneNetDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line
    and converts it into a ``Dataset`` suitable for sequence tagging. 

    Parameters
    ----------
    """
    def __init__(self,
                 token_delimiter: str = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._token_delimiter = token_delimiter

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        if file_path.endswith("zip"):
            archive = zipfile.ZipFile(file_path, "r")
            data_file = archive.open(os.path.basename(file_path)[:-4])
        else:
            data_file = open(file_path, "r")

        logger.info("Reading instances from lines in file at: %s", file_path)

        dialogs = json.load(data_file)

        for dial_name in dialogs:
            dialog = dialogs[dial_name]["log"]
            for turn in dialog:
                tokens = turn["text"].split()
                spans = turn["span_info"]
                tags = []
                domain = "None"
                intent = "None"
                for i in range(len(tokens)):
                    for span in spans:
                        if i == span[3]:
                            new_domain, new_intent = span[0].split("-", 1)
                            if domain == "None":
                                domain = new_domain
                            elif domain != new_domain:
                                continue
                            if intent == "None":
                                intent = new_intent
                            elif intent != new_intent:
                                continue
                            tags.append("B-"+span[1])
                            break
                        if i > span[3] and i <= span[4]:
                            new_domain, new_intent = span[0].split("-", 1)
                            if domain != new_domain:
                                continue
                            if intent != new_intent:
                                continue
                            tags.append("I-"+span[1])
                            break
                    else:
                        tags.append("O")

                if domain != "None":
                    assert intent != "None", "intent must not be None when domain is not None"
                elif turn["dialog_act"] != {}:
                    assert intent == "None", "intent must be None when domain is None"
                    di = list(turn["dialog_act"].keys())[0]
                    dai = turn["dialog_act"][di][0]
                    domain = di.split("-")[0]
                    intent = di.split("-", 1)[-1] + "+" + dai[0] + "*" + dai[1]

                dialog_act = {}
                for dacts in turn["span_info"]:
                    if dacts[0] not in dialog_act:
                        dialog_act[dacts[0]] = []
                    dialog_act[dacts[0]].append([dacts[1], " ".join(tokens[dacts[3]: dacts[4]+1])])

                for dacts in turn["dialog_act"]:
                    for dact in turn["dialog_act"][dacts]:
                        if dacts not in dialog_act:
                            dialog_act[dacts] = turn["dialog_act"][dacts]
                            break
                        elif dact[0] not in [sv[0] for sv in dialog_act[dacts]]:
                            dialog_act[dacts].append(dact)

                tokens = [Token(token) for token in tokens]

                yield self.text_to_instance(tokens, tags, domain, intent, dialog_act)


    def text_to_instance(self, tokens: List[Token], tags: List[str] = None, domain: str = None,
        intent: str = None, dialog_act: Dict[str, Any] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        if tags:
            fields["tags"] = SequenceLabelField(tags, sequence)
        if domain:
            fields["domain"] = LabelField(domain, label_namespace="domain_labels")
        if intent:
            fields["intent"] = LabelField(intent, label_namespace="intent_labels")
        if dialog_act is not None:
            fields["metadata"] = MetadataField({"words": [x.text for x in tokens],
            'dialog_act': dialog_act})
        else:
            fields["metadata"] = MetadataField({"words": [x.text for x in tokens], 'dialog_act': {}})
        return Instance(fields)
