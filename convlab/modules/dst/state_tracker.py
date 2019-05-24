# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

class Tracker:
    """Base class for dialog state tracker models."""
    def __init__(self):
        """The constructor of Tracker class."""
        pass

    def update(self, user_act=None):
        """
        Update dialog state based on new user dialog act.
        Args:
            sess (Session Object): (for models implemented using tensorflow) The Session Object to assist model running.
            user_act (dict or str): The dialog act (or utterance) of user input. The class of user_act depends on
                    the method of state tracker. For example, for rule-based tracker, type(user_act) == dict; while for
                    MDBT, type(user_act) == str.
        Returns:
            new_state (dict): Updated dialog state, with the same form of previous state. Note that the dialog state is
                    also a private data member.
        """
        pass

    def init_session(self):
        """Init the Tracker to start a new session."""
        pass