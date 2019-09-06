import json
import logging
import math
import os
import zipfile
from typing import Dict

import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, Field
from allennlp.data.instance import Instance
from overrides import overrides

from convlab.lib.file_util import cached_path
from convlab.modules.dst.multiwoz.rule_dst import RuleDST
from convlab.modules.action_decoder.multiwoz.multiwoz_vocab_action_decoder import ActionVocab
from convlab.modules.state_encoder.multiwoz.multiwoz_state_encoder import MultiWozStateEncoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("mle_policy")
class MlePolicyDatasetReader(DatasetReader):
    """
    Reads instances from a data file:

    Parameters
    ----------
    """
    def __init__(self,
                 num_actions: int,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.dst = RuleDST()
        self.action_vocab = ActionVocab(num_actions=num_actions)
        self.action_list = self.action_vocab.vocab
        self.state_encoder = MultiWozStateEncoder()

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
            self.dst.init_session()
            for i, turn in enumerate(dialog):
                if i % 2 == 0:  # user turn
                    self.dst.update(user_act=turn["dialog_act"])
                else:  # system turn
                    delex_act = {}
                    for domain_act in turn["dialog_act"]:
                        domain, act_type = domain_act.split('-', 1)
                        if act_type in ['NoOffer', 'OfferBook']:
                            delex_act[domain_act] = ['none']
                        elif act_type in ['Select']:
                            for sv in turn["dialog_act"][domain_act]:
                                if sv[0] != "none":
                                    delex_act[domain_act] = [sv[0]]
                                    break
                        else:
                            delex_act[domain_act] = [sv[0] for sv in turn["dialog_act"][domain_act]]
                    state_vector = self.state_encoder.encode(self.dst.state)
                    action_index = self.find_best_delex_act(delex_act)

                    yield self.text_to_instance(state_vector, action_index)

    def find_best_delex_act(self, action):
        def _score(a1, a2):
            score = 0
            for domain_act in a1:
                if domain_act not in a2:
                    score += len(a1[domain_act])
                else:
                    score += len(set(a1[domain_act]) - set(a2[domain_act]))
            return score

        best_p_action_index = -1
        best_p_score = math.inf
        best_pn_action_index = -1
        best_pn_score = math.inf
        for i, v_action in enumerate(self.action_list):
            if v_action == action:
                return i
            else:
                p_score = _score(action, v_action)
                n_score = _score(v_action, action)
                if p_score > 0 and n_score == 0 and p_score < best_p_score:
                    best_p_action_index = i
                    best_p_score = p_score
                else:
                    if p_score + n_score < best_pn_score:
                        best_pn_action_index = i
                        best_pn_score = p_score + n_score
        if best_p_action_index >= 0:
            return best_p_action_index
        return best_pn_action_index

    def text_to_instance(self, state: np.ndarray, action: int = None) -> Instance:  # type: ignore
        """
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields["states"] = ArrayField(state)
        if action is not None:
            fields["actions"] = LabelField(action, skip_indexing=True)
        return Instance(fields)
