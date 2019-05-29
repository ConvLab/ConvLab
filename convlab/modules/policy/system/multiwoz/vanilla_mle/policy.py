"""
"""

import os
from pprint import pprint
import numpy as np

from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.data import DatasetReader

from convlab.modules.policy.system.policy import SysPolicy
from convlab.modules.policy.system.multiwoz.util import state_encoder, action_decoder

DEFAULT_CUDA_DEVICE=-1
DEFAULT_ARCHIVE_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/300/model.tar.gz")


class VanillaMlePolicy(SysPolicy):
    """Vanilla MLE trained policy."""

    def __init__(self, 
                archive_file=DEFAULT_ARCHIVE_FILE,
                cuda_device=DEFAULT_CUDA_DEVICE):
        """ Constructor for NLU class. """
        SysPolicy.__init__(self)
        check_for_gpu(cuda_device)
        archive = load_archive(archive_file,
                            cuda_device=cuda_device)
        dataset_reader_params = archive.config["dataset_reader"]
        self.dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.action_vocab = self.dataset_reader.action_vocab 
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
        state_vector = state_encoder(state)
        # pprint(np.nonzero(state_vector))

        instance = self.dataset_reader.text_to_instance(state_vector)
        outputs = self.model.forward_on_instance(instance)
        # print(outputs["actions"])
        dialacts = action_decoder(state, outputs["actions"], self.action_vocab)
        # print(outputs["probs"])
        if dialacts == {'general-bye': [['none', 'none']]}:
            # print(outputs["probs"][outputs["actions"]])
            outputs["probs"][outputs["actions"]] = 0
            outputs["actions"] = np.argmax(outputs["probs"])
            # print(outputs["actions"])
            # print(outputs["probs"][outputs["actions"]])
            dialacts = action_decoder(state, outputs["actions"], self.action_vocab)


        if state == init_state():
            dialacts = {}

        return dialacts 


if __name__ == "__main__":
    from convlab.modules.dst.multiwoz.dst_util import init_state

    policy = VanillaMlePolicy()
    pprint(policy.predict(init_state()))
