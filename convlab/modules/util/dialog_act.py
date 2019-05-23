"""
"""

class DialogAct:
    """The base class for all kinds of dialog act."""
    def __init__(self):
        pass


class DeterministicAct(DialogAct):
    """Deterministic dialog act class."""
    def __init__(self, act_type, sv_pairs):
        """
        Constructor for deterministic dialog act class.
        Args:
            act_type (str): A dialog act type name.
            sv_pairs (list): A list of slot-value pairs, e.g., [['location', 'north'], ['price', 'cheap']]
        """
        DialogAct.__init__(self)
        self.act_type = act_type
        self.sv_pairs = sv_pairs
        self.available_slots = [item[0] for item in self.sv_pairs]


class StochasticAct(DialogAct):
    """Stochastic dialog act class."""
    def __init__(self, act_type_vec, sv_pair_vecs):
        """
        Constructor for stochastic dialog act class.
        Args:
            act_type_vec (list): A prob distribution over all act types, where len(act_type_vec) = |act_types|.
            sv_pair_vecs (list): The prob distributions over the values of each slot, where each item is a list. The
                    length of slot X's item is |X_values|+1 where the extra 1 dimension indicates X is not mentioned or
                    dont_care.
        """
        DialogAct.__init__(self)
        self.act_type_vec = act_type_vec
        self.sv_pair_vecs = sv_pair_vecs