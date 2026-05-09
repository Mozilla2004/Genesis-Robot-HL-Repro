"""Random baseline policy."""

import numpy as np


class RandomPolicy:
    """Random action policy for baseline comparison."""

    def __init__(self, env):
        """
        Initialize random policy.

        Args:
            env: Gymnasium environment with action_space
        """
        self.action_space = env.action_space
        self.action_dim = self.action_space.shape[0]

    def reset(self):
        """Reset policy state (no-op for random policy)."""
        pass

    def act(self, obs):
        """
        Return random action from action space.

        Args:
            obs: Observation (not used for random policy)

        Returns:
            action: Random action sampled from action_space
        """
        return self.action_space.sample()
