# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

class NLG:
    """Base class for NLG model."""
    def __init__(self):
        """ Constructor for NLG class. """
        pass

    def generate(self, dialog_act):
        """
        Generate a natural language utterance conditioned on the dialog act produced by Agenda or Policy.
        Args:
            dialog_act (dict): The dialog act of the following system response. The dialog act can be either produced
                    by user agenda or system policy module.
        Returns:
            response (str): The natural language utterance of the input dialog_act.
        """
        pass