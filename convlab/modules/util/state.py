"""
"""

class State:
    """Base class for dialog state."""
    def __init__(self):
        """
        Construct a init state variable for the class.
        The dialog state is a dict type variable, including the necessary aspects as shown in following code.
        Variable:
        Action (dict): System/User act, with the form of {act_type1: [[slot_name_1, value_1], [slot_name_2, value_2], ...], ...}
        Current_slot
        """
        state = {
            'user_action': None,
            'current_slots': None,
            'kb_results_dict': None,
            'turn': None,
            'history': None,
            'agent_action': None
        }
        self.state = state

    def set(self, new_state):
        """
        Set the state with the new_state variable.
        Args:
            new_state (dict): A new state variable.
        """
        self.state = new_state

    def set_aspect(self, slot, value):
        """
        Set the value for certrain slot.
        Args:
            slot (str): The name of slot.
            value: The value of the slot.
        """
        self.state[slot] = value