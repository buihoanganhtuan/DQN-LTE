from __future__ import division
import numpy as np

from rl2.util import *

class Policy(object):
    """Abstract base class for all implemented policies.
    Each policy helps with selection of action to take on an environment.
    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods:
    - `select_action`
    # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        """Return configuration of the policy
        # Returns
            Configuration as dict
        """
        return {}

# We implement specific types of policy below. The LinearAnnealedPolicy is 
# actually an outer policy that is in charge of changing, says, eps of inner
# policies such as EpsGreedyQPolicy
class LinearAnnealedPolicy(Policy):
    """Implement the linear annealing policy
    Linear Annealing Policy computes a current threshold value and
    transfers it to an inner policy which chooses the action. The threshold
    value is following a linear function decreasing over time."""
    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError('Policy does not have attribute "{}".'.format(attr))

        super().__init__()

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps

    def get_current_value(self):
        """Return current annealing value
        # Returns
            Value to use in annealing
        """
        if self.agent.training:
            # Linear annealed: f(x) = ax + b.
            a = -float(self.value_max - self.value_min) / float(self.nb_steps)
            b = float(self.value_max)
            value = max(self.value_min, a * float(self.agent.step) + b)
        else:
            value = self.value_test
        return value

    def select_action(self, **kwargs):
        """Choose an action to perform
        # Returns
            Action to take (int)
        """
        setattr(self.inner_policy, self.attr, self.get_current_value())
        return self.inner_policy.select_action(**kwargs)

    @property
    def metrics_names(self):
        """Return names of metrics
        # Returns
            List of metric names
        """
        return ['mean_{}'.format(self.attr)]        

    @property
    def metrics(self):
        """Return metrics values
        # Returns
            List of metric values
        """     
        return [getattr(self.inner_policy, self.attr)]

    def get_config(self):
        """Return configurations of LinearAnnealedPolicy
        # Returns
            Dict of config
        """
        config = super().get_config()
        config['attr'] = self.attr
        config['value_max'] = self.value_max
        config['value_min'] = self.value_min
        config['value_test'] = self.value_test
        config['nb_steps'] = self.nb_steps
        # the get_object_config() method is defined in util.py
        config['inner_policy'] = get_object_config(self.inner_policy)

class EpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy
    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=1.):
        super().__init__()
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy
        # Returns
            Dict of config
        """
        config = super().get_config()
        config['eps'] = self.eps
        return config                

class GreedyQPolicy(Policy):
    """Implement the greedy policy
    Greedy policy returns the current best action according to q_values
    """
    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action            