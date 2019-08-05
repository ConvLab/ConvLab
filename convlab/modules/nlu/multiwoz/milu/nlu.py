# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import os
from pprint import pprint

from allennlp.common.checks import check_for_gpu
from allennlp.data import DatasetReader
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models.archival import load_archive

from convlab.lib.file_util import cached_path
from convlab.modules.nlu.nlu import NLU
from convlab.modules.nlu.multiwoz.milu import dataset_reader, model 

from spacy.symbols import ORTH, LEMMA

DEFAULT_CUDA_DEVICE = -1
DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "milu.tar.gz")

class MILU(NLU):
    """Multi-intent language understanding model."""

    def __init__(self,
                archive_file=DEFAULT_ARCHIVE_FILE,
                cuda_device=DEFAULT_CUDA_DEVICE,
                model_file=None,
                context_size=3):
        """ Constructor for NLU class. """

        self.context_size = context_size

        check_for_gpu(cuda_device)

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for MILU is specified!")

            archive_file = cached_path(model_file)

        archive = load_archive(archive_file,
                            cuda_device=cuda_device)
        self.tokenizer = SpacyWordSplitter(language="en_core_web_sm")
        _special_case = [{ORTH: u"id", LEMMA: u"id"}]
        self.tokenizer.spacy.tokenizer.add_special_case(u"id", _special_case)

        dataset_reader_params = archive.config["dataset_reader"]
        self.dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.model = archive.model
        self.model.eval()


    def parse(self, utterance, context=[]):
        """
        Predict the dialog act of a natural language utterance and apply error model.
        Args:
            utterance (str): A natural language utterance.
        Returns:
            output (dict): The dialog act of utterance.
        """
        if len(utterance) == 0:
            return {}

        if self.context_size > 0 and len(context) > 0:
            context_tokens = sum([self.tokenizer.split_words(utterance+" SENT_END") for utterance in context[-self.context_size:]], [])
        else:
            context_tokens = self.tokenizer.split_words("SENT_END")
        tokens = self.tokenizer.split_words(utterance)
        instance = self.dataset_reader.text_to_instance(context_tokens, tokens)
        outputs = self.model.forward_on_instance(instance)

        return outputs["dialog_act"]


if __name__ == "__main__":
    nlu = MILU()
    test_contexts = [
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
    ]
    test_utterances = [
        "What type of accommodations are they. No , i just need their address . Can you tell me if the hotel has internet available ?",
        "What type of accommodations are they.",
        "No , i just need their address .",
        "Can you tell me if the hotel has internet available ?",
        "you're welcome! enjoy your visit! goodbye.",
        "yes. it should be moderately priced.",
        "i want to book a table for 6 at 18:45 on thursday",
        "i will be departing out of stevenage.",
        "What is the Name of attraction ?",
        "Can I get the name of restaurant?",
        "Can I get the address and phone number of the restaurant?",
        "do you have a specific area you want to stay in?"
    ]
    for ctxt, utt in zip(test_contexts, test_utterances):
        print(ctxt)
        print(utt)
        pprint(nlu.parse(utt))
        # pprint(nlu.parse(utt.lower()))

    test_contexts = [
        "The phone number of the hotel is 12345678",
        "I have many that meet your requests",
        "The phone number of the hotel is 12345678",
        "I found one hotel room",
        "thank you",
        "Is it moderately priced?",
        "Can I help you with booking?",
        "Where are you departing from?",
        "I found an attraction",
        "I found a restaurant",
        "I found a restaurant",
        "I'm looking for a place to stay.",
    ]
    for ctxt, utt in zip(test_contexts, test_utterances):
        print(ctxt)
        print(utt)
        pprint(nlu.parse(utt, [ctxt]))
        # pprint(nlu.parse(utt.lower(), ctxt.lower()))
