"""
"""

import os
from pprint import pprint

import numpy as np
from allennlp.common.checks import check_for_gpu
from allennlp.data import DatasetReader
from allennlp.models.archival import load_archive

from convlab.lib.file_util import cached_path
from convlab.modules.dst.multiwoz.dst_util import init_state
from convlab.modules.action_decoder.multiwoz.multiwoz_vocab_action_decoder import MultiWozVocabActionDecoder
from convlab.modules.policy.system.policy import SysPolicy

DEFAULT_CUDA_DEVICE=-1
DEFAULT_ARCHIVE_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/300/model.tar.gz")


class VanillaMLEPolicy(SysPolicy):
    """Vanilla MLE trained policy."""

    def __init__(self, 
                archive_file=DEFAULT_ARCHIVE_FILE,
                cuda_device=DEFAULT_CUDA_DEVICE,
                model_file=None):
        """ Constructor for NLU class. """
        SysPolicy.__init__(self)

        check_for_gpu(cuda_device)

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for MILU is specified!")
            archive_file = cached_path(model_file)

        archive = load_archive(archive_file,
                            cuda_device=cuda_device)
        dataset_reader_params = archive.config["dataset_reader"]
        self.dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.action_decoder = MultiWozVocabActionDecoder()
        self.action_decoder.action_vocab = self.dataset_reader.action_vocab
        self.state_encoder = self.dataset_reader.state_encoder
        self.model = archive.model
        self.model.eval()

    def predict(self, state):
        """
        Predict the dialog act of a natural language utterance and apply error model.
        Args:
            utterance (str): A natural language utterance.
        Returns:
            output (dict): The dialog act of utterance.
        """
        state_vector = self.state_encoder.encode(state)

        instance = self.dataset_reader.text_to_instance(state_vector)
        outputs = self.model.forward_on_instance(instance)
        dialacts = self.action_decoder.decode(outputs["actions"], state)
        if dialacts == {'general-bye': [['none', 'none']]}:
            outputs["probs"][outputs["actions"]] = 0
            outputs["actions"] = np.argmax(outputs["probs"])
            dialacts = self.action_decoder.decode(outputs["actions"], state)


        if state == init_state():
            dialacts = {}

        return dialacts 


if __name__ == "__main__":
    policy = VanillaMLEPolicy()
    pprint(policy.predict(init_state()))
