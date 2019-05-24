# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

class NLU:
    """Base class for NLU model."""

    def __init__(self):
        """ Constructor for NLU class. """

    def parse(self, utterance, context=None):
        """
        Predict the dialog act of a natural language utterance and apply error model.
        Args:
            utterance (str): The user input, a natural language utterance.
        Returns:
            output (dict): The parsed dialog act of the input NL utterance.
        """
        pass