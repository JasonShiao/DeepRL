import torch
import torch.nn as nn
from rl_physics.infrastructure import pytorch_util as ptu
#import numpy as np

class MLPBasicCritic(nn.Module):
    """
    Critic network for TD3 algorithm.
    This network is used to estimate the Q-value of a given state-action pair.
    """

    def __init__(self, ob_dim, ac_dim, n_layers=2, size=256, learning_rate=1e-3):
        super(MLPBasicCritic, self).__init__()

        # Define the input size
        input_size = ob_dim + ac_dim

        # Create the layers of the network
        q1_layers = [
            nn.Linear(input_size, size), nn.ReLU(),
            *(nn.Linear(size, size), nn.ReLU()) * (n_layers - 1),
            nn.Linear(size, 1)
        ]
        self.q1_model = nn.Sequential(*q1_layers)
                
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        self.to(ptu.device)
    
    def forward(self, obs, acs):
        """
        Forward pass through the network.

        :param obs: Observation tensor
        :param acs: Action tensor
        :return: Q-value tensor
        """
        obs_acs = torch.cat([obs, acs], dim=-1)
        q1 = self.q1_model(obs_acs)
        return q1

    def Q1(self, obs, acs):
        """
        Get the Q-value of the first critic network.

        :param
        """
        obs_acs = torch.cat([obs, acs], dim=-1)
        q1 = self.q1_model(obs_acs)
        return q1
    
    def save(self, filepath):
        """
        Save the model to a file.
        """
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """
        Load the model from a file.
        """
        self.load_state_dict(torch.load(filepath))
        self.eval()    


class MLPTwinCritic(nn.Module):
    """
    Critic network for TD3 algorithm.
    This network is used to estimate the Q-value of a given state-action pair.
    """

    def __init__(self, ob_dim, ac_dim, n_layers=2, size=256, learning_rate=1e-3):
        super(MLPTwinCritic, self).__init__()

        # Define the input size
        input_size = ob_dim + ac_dim

        # Create the layers of the network
        q1_layers = [
            nn.Linear(input_size, size), nn.ReLU(),
            *(nn.Linear(size, size), nn.ReLU()) * (n_layers - 1),
            nn.Linear(size, 1)
        ]
        self.q1_model = nn.Sequential(*q1_layers)
        
        q2_layers = [
            nn.Linear(input_size, size), nn.ReLU(),
            *(nn.Linear(size, size), nn.ReLU()) * (n_layers - 1),
            nn.Linear(size, 1)
        ]
        self.q2_model = nn.Sequential(*q2_layers)
        
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        self.to(ptu.device)
    
    #def get_q_value(self, obs, acs):
    #    """
    #    Get the Q-value (always in batch mode) for a given observation and action.
    #    
    #    :param obs: Observation numpy array
    #    :param acs: Action numpy array
    #    :return: Q-value tensor
    #    """
    #    obs_tensor = ptu.from_numpy(np.atleast_2d(obs))
    #    acs_tensor = ptu.from_numpy(np.atleast_2d(acs))
    #    q1_value, q2_value = self(obs_tensor, acs_tensor)
    #    return ptu.to_numpy(q1_value), ptu.to_numpy(q2_value)

    def forward(self, obs, acs):
        """
        Forward pass through the network.

        :param obs: Observation tensor
        :param acs: Action tensor
        :return: Q-value tensor
        """
        obs_acs = torch.cat([obs, acs], dim=-1)
        q1 = self.q1_model(obs_acs)
        q2 = self.q2_model(obs_acs)
        return q1, q2

    def Q1(self, obs, acs):
        """
        Get the Q-value of the first critic network.

        :param
        """
        obs_acs = torch.cat([obs, acs], dim=-1)
        q1 = self.q1_model(obs_acs)
        return q1
    
    def save(self, filepath):
        """
        Save the model to a file.
        """
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """
        Load the model from a file.
        """
        self.load_state_dict(torch.load(filepath))
        self.eval()    