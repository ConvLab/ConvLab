"""
The policy base class for system bot.
"""


class SysPolicy:
    """Base class for system policy model."""

    def __init__(self):
        """ Constructor for SysPolicy class. """
        pass

    def predict(self, state):
        """
        Predict the system action (dialog act) given state.
        Args:
            state (dict): Dialog state. For more details about the each field of the dialog state, please refer to
                    the init_state method in convlab/dst/dst_util.py
        Returns:
            action (dict): The dialog act of the current turn system response, which is then passed to NLG module to
                    generate a NL utterance.
        """
        pass
    
    def init_session(self):
        """Init the SysPolicy module to start a new session."""
        pass
