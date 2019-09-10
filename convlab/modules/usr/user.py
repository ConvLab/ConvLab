# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from convlab.modules.util.multiwoz.da_normalize import da_normalize

"""
"""

class UserSimulator:
    """An aggregation of user simulator components."""
    def __init__(self, nlu_model, policy, nlg_model):
        """
        The constructor of UserSimulator class. The input are the models of each component.
        Args:
            nlu_model (NLU): An instance of NLU class.
            policy (UserPolicy): An instance of Policy class.
            nlg_model (NLG): An instance of NLG class.
        """
        self.nlu_model = nlu_model
        # self.tracker = tracker
        self.policy = policy
        self.nlg_model = nlg_model

        self.sys_act = None
        self.current_action = None
        self.policy.init_session()

    def response(self, input, context=[]):
        """
        Generate the response of user.
        Args:
            input (str or dict): Preorder system output. The type is str if system.nlg is not None, else dict.
        Returns:
            output (str or dict): User response. If the nlg component is None, type(output) == dict, else str.
            action (dict): The dialog act of output. Note that if the nlg component is None, the output and action are
                    identical.
            session_over (boolean): True to terminate session, else session continues.
            reward (float): The reward given by the user.
        """

        if self.nlu_model is not None:
            sys_act = self.nlu_model.parse(input, context)
            sys_act = da_normalize(sys_act, role='sys')
        else:
            sys_act = input
        self.sys_act = sys_act
        action, session_over, reward = self.policy.predict(None, sys_act)
        if self.nlg_model is not None:
            output = self.nlg_model.generate(action)
        else:
            output = action

        self.current_action = action

        return output, action, session_over, reward

    def init_session(self):
        """Init the parameters for a new session by calling the init_session methods of policy component."""
        self.policy.init_session()
        self.current_action = None

    def init_response(self):
        """Return a init response of the user."""
        if self.nlg_model is not None:
            output = self.nlg_model.generate({})
        else:
            output = {}
        return output