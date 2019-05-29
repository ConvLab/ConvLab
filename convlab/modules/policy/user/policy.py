"""
The policy base class for user bot.
"""


class UserPolicy:
    """Base model for user policy model."""
    def __init__(self):
        """ Constructor for UserPolicy class. """
        pass

    def predict(self, state, sys_action):
        """
        Predict an user act based on state and preorder system action.
        Args:
            state (tuple): Dialog state.
            sys_action (tuple): Preorder system action.s
        Returns:
            action (tuple): User act.
            session_over (boolean): True to terminate session, otherwise session continues.
            reward (float): Reward given by the user.
        """
        pass

    def init_session(self):
        """
        Restore after one session
        """
        pass