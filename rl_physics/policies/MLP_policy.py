import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch import optim

#import numpy as np
import torch
#from torch import distributions

from rl_physics.infrastructure import pytorch_util as ptu
from rl_physics.policies.base_policy import BasePolicy

from torch.distributions.distribution import Distribution


# Actor network
class MLP_Pi_FC(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, ac_dim, ob_dim, n_layers=2, size=256,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs):
        super(MLP_Pi_FC, self).__init__()
        
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        
        layers = [nn.Linear(ob_dim, size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(size, size), nn.ReLU()]
        self.fc = nn.Sequential(*layers)
        #self.fc1 = torch.nn.Linear(obs_size, 256)
        #self.fc2 = torch.nn.Linear(256, 256)
        self.mu = torch.nn.Linear(size, ac_dim)
        self.log_sigma = torch.nn.Linear(size, ac_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(ptu.device)

    def forward(self, observation: torch.FloatTensor, deterministic=False, with_logprob=False):
        #y1 = F.relu(self.fc1(x))
        #y2 = F.relu(self.fc2(y1))
        y = self.fc(observation)
        mu = self.mu(y)

        if deterministic:
            # used for evaluating policy
            action = torch.tanh(mu)
            log_prob = None
        else:
            log_sigma = self.log_sigma(y)
            log_sigma = torch.clamp(log_sigma,min=-20.0,max=2.0)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            x_t = dist.rsample()
            action = torch.tanh(x_t) # map to [-1, 1]
            if with_logprob:
                log_prob = dist.log_prob(x_t).sum(1)
                # Tanh correction
                log_prob -= torch.log(torch.clamp(1-action.pow(2),min=1e-6)).sum(1)
            else:
                log_prob = None

        return action, log_prob
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

class MLPStochasticPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    #def get_action(self, obs: np.ndarray) -> np.ndarray:
    #    observation = np.atleast_2d(obs)
    #
    #    # TODO return the action that the policy prescribes
    #    # Note that the default policy above defines parameters for both mean and variance.
    #    # It is up to you whether you want to use both to sample actions (recommended) or just the mean.
    #    with torch.no_grad():
    #        assert True, "get_action Not implemented yet"
    #        TODO: ....
    #        ac_distrib = self.forward(ptu.from_numpy(observation))
    #        ac = ptu.to_numpy(ac_distrib.sample())
    #    return ac

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Distribution:
        std = torch.exp(self.logstd)
        return torch.distributions.Normal(self.mean_net(observation), std)

#####################################################

class MLPDeterministicPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            # TBA:
            pass
            #self.logits_na = ptu.build_mlp(
            #    input_size=self.ob_dim,
            #    output_size=self.ac_dim,
            #    n_layers=self.n_layers,
            #    size=self.size,
            #)
            #self.logits_na.to(ptu.device)
            #self.mean_net = None
            #self.logstd = None
            #self.optimizer = optim.Adam(self.logits_na.parameters(),
            #                            self.learning_rate)
        else:
            # TODO: Add Tanh activation to the last layer and scale the output based on action bounds
            layers = [
                nn.Linear(self.ob_dim, self.size), nn.ReLU(),
                *(nn.Linear(self.size, self.size), nn.ReLU()) * (self.n_layers - 1),
                nn.Linear(self.size, self.ac_dim),
                nn.Tanh(),
            ]
            self.model = nn.Sequential(*layers)
            self.model.to(ptu.device)
            self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    #def get_action(self, obs: np.ndarray) -> np.ndarray:
    #    observation = np.atleast_2d(obs)
    #
    #    # TODO return the action that the policy prescribes
    #    # Note that the default policy above defines parameters for both mean and variance.
    #    # It is up to you whether you want to use both to sample actions (recommended) or just the mean.
    #    with torch.no_grad():
    #        assert True, "get_action Not implemented yet"
    #        TODO: ....
    #        ac_distrib = self.forward(ptu.from_numpy(observation))
    #        ac = ptu.to_numpy(ac_distrib.sample())
    #    return ac

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(observation)
