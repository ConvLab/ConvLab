# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

class ErrorNLU:
    """Base model for generating NLU error."""
    def __init__(self, act_type_rate=0.0, slot_rate=0.0):
        """
        Args:
            act_type_rate (float): The error rate applied on dialog act type.
            slot_rate (float): Error rate applied on slots.
        """
        self.set_error_rate(act_type_rate, slot_rate)

    def set_error_rate(self, act_type_rate, slot_rate):
        """
        Set error rate parameter for error model.
        Args:
            act_type_rate (float): The error rate applied on dialog act type.
            slot_rate (float): Error rate applied on slots.
        """
        self.act_type_rate = act_type_rate
        self.slot_rate = slot_rate

    def apply(self, dialog_act):
        """
        Apply the error model on dialog act.
        Args:
            dialog_act (tuple): Dialog act.
        Returns:
            dialog_act (tuple): Dialog act with noise.
        """
        #TODO
        return
