from rl_physics.env_models.base_env_model import BaseEnvModel
from rl_physics.env_models.base_env_model import MLPDeterministicEnv, LNNDeterministicEnv
import numpy as np

class PendulumEnvModel(BaseEnvModel):
    """
    A simple environment model that predicts the next state and reward
    using a linear model.
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the Pendulum environment model.

        Args:
            state_dim: The dimension of the state space.
            action_dim: The dimension of the action space.
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nn_model = MLPDeterministicEnv(action_dim, state_dim, 3, 64, learning_rate=1e-3)
        #self.nn_model = LNNDeterministicEnv("pendulum", 1, state_dim, action_dim)

    def step(self, state, action):
        """
        Predict the next state and reward given the current state and action.
        Can be batch or single input

        Args:
            state (np.ndarray): The current state of the environment. Shape (3,) or (batch_size, 3)
            action (np.ndarray): The action taken by the agent. Shape (1,) or (batch_size, 1)

        Returns: (Always in batch format)
            next_state (np.ndarray): The predicted next state. Shape (batch_size, 3)
            reward (np.ndarray): The predicted reward. Shape (batch_size,)
        """
        # Predict next state from model
        next_state_batch, reward_batch = self.nn_model.get_next_state(state, action)  # shape (batch_size, 3)

        # Compute reward based on predicted next state and action
        #cos_th = next_state_batch[:, 0]
        #sin_th = next_state_batch[:, 1]
        #th = np.arctan2(sin_th, cos_th)  # angle in radians
        #thdot = next_state_batch[:, 2]         # angular velocity
        #u = np.atleast_1d(action).reshape(-1)  # make sure action is 1D (batch,)

        # Reward is negative cost
        #costs_batch = self._angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        #reward_batch = -costs_batch

        return next_state_batch, reward_batch
    
    #def _angle_normalize(self, x):
    #    return ((x + np.pi) % (2 * np.pi)) - np.pi
    
    def train(self, states, actions, rewards, next_states):
        """
        Train the environment model using the provided data.

        Args:
            states: The current states of the environment.
            actions: The actions taken by the agent.
            rewards: The rewards received.
            next_states: The predicted next states.
        """
        return self.nn_model.update(states, actions, rewards, next_states)