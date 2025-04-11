import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
#from torch.distributions.normal import Normal
from torch.func import jacrev
#from collections import deque
#import random

import numpy as np
import torch
from torch import distributions
#torch.set_default_dtype(torch.float64)

from rl_physics.infrastructure import pytorch_util as ptu
from torch.distributions.distribution import Distribution


# LNN dynamics model
class LNNDeterministicEnv(torch.nn.Module):
    def __init__(self, env_name, n, obs_size, action_size, dt=0.02, a_zeros=0):
        super(LNNDeterministicEnv, self).__init__()
        self.env_name = env_name
        self.dt = dt
        self.n = 1

        self.reward_net = nn.Sequential(
            nn.Linear(obs_size + action_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )

        input_size = 2
        out_L = int(self.n*(self.n+1)/2)
        self.fc1_L = torch.nn.Linear(input_size, 64)
        self.fc2_L = torch.nn.Linear(64, 64)
        self.fc3_L = torch.nn.Linear(64, out_L)
        if self.env_name != "reacher":
            self.fc1_V = torch.nn.Linear(input_size, 64)
            self.fc2_V = torch.nn.Linear(64, 64)
            self.fc3_V = torch.nn.Linear(64, 1)
            params = itertools.chain(
                self.fc1_L.parameters(), self.fc2_L.parameters(), self.fc3_L.parameters(),
                self.fc1_V.parameters(), self.fc2_V.parameters(), self.fc3_V.parameters(),
                self.reward_net.parameters()
            )
        else:
            params = itertools.chain(
                self.fc1_L.parameters(), self.fc2_L.parameters(), self.fc3_L.parameters(),
                self.reward_net.parameters()
            )

        self.a_zeros = a_zeros
        
        self.optimizer = optim.Adam(
            params,
            lr=1e-3
        )
        self.to(ptu.device)

    def trig_transform_q(self, q):
        if self.env_name == "pendulum":
            return torch.column_stack((torch.cos(q[:,0]),torch.sin(q[:,0])))
        
        elif self.env_name == "reacher" or self.env_name == "acrobot":
            return torch.column_stack((torch.cos(q[:,0]),torch.sin(q[:,0]),\
                                       torch.cos(q[:,1]),torch.sin(q[:,1])))
        
        elif self.env_name == "cartpole":
            return torch.column_stack((q[:,0],\
                                       torch.cos(q[:,1]),torch.sin(q[:,1])))
        
        elif self.env_name == "cart2pole":
            return torch.column_stack((q[:,0],\
                                       torch.cos(q[:,1]),torch.sin(q[:,1]),\
                                       torch.cos(q[:,2]),torch.sin(q[:,2])))

        elif self.env_name == "cart3pole":
            return torch.column_stack((q[:,0],\
                                       torch.cos(q[:,1]),torch.sin(q[:,1]),\
                                       torch.cos(q[:,2]),torch.sin(q[:,2]),\
                                       torch.cos(q[:,3]),torch.sin(q[:,3])))
        
        elif self.env_name == "acro3bot":
            return torch.column_stack((torch.cos(q[:,0]),torch.sin(q[:,0]),\
                                       torch.cos(q[:,1]),torch.sin(q[:,1]),\
                                       torch.cos(q[:,2]),torch.sin(q[:,2])))

    def inverse_trig_transform_model(self, x):
        if self.env_name == "pendulum":
            return torch.cat((torch.atan2(x[:,1],x[:,0]).unsqueeze(1),x[:,2:]),1)
        
        elif self.env_name == "reacher" or self.env_name == "acrobot":
            return torch.cat((torch.atan2(x[:,1],x[:,0]).unsqueeze(1),torch.atan2(x[:,3],x[:,2]).unsqueeze(1),x[:,4:]),1)
        
        elif self.env_name == "cartpole":
            return torch.cat((x[:,0].unsqueeze(1),torch.atan2(x[:,2],x[:,1]).unsqueeze(1),x[:,3:]),1)
        
        elif self.env_name == "cart2pole":
            return torch.cat((x[:,0].unsqueeze(1),torch.atan2(x[:,2],x[:,1]).unsqueeze(1),torch.atan2(x[:,4],x[:,3]).unsqueeze(1),x[:,5:]),1)

        elif self.env_name == "cart3pole":
            return torch.cat((x[:,0].unsqueeze(1),torch.atan2(x[:,2],x[:,1]).unsqueeze(1),torch.atan2(x[:,4],x[:,3]).unsqueeze(1),
                              torch.atan2(x[:,6],x[:,5]).unsqueeze(1),x[:,7:]),1)

        elif self.env_name == "acro3bot":
            return torch.cat((torch.atan2(x[:,1],x[:,0]).unsqueeze(1),torch.atan2(x[:,3],x[:,2]).unsqueeze(1),torch.atan2(x[:,5],x[:,4]).unsqueeze(1),
                              x[:,6:]),1)

    def compute_L(self, q):
        y1_L = F.softplus(self.fc1_L(q))
        y2_L = F.softplus(self.fc2_L(y1_L))
        y_L = self.fc3_L(y2_L)
        device = y_L.device
        if self.n == 1:
            L = y_L.unsqueeze(1)

        elif self.n == 2:
            L11 = y_L[:,0].unsqueeze(1)
            L1_zeros = torch.zeros(L11.size(0),1, dtype=torch.float64, device=device)

            L21 = y_L[:,1].unsqueeze(1)
            L22 = y_L[:,2].unsqueeze(1)

            L1 = torch.cat((L11,L1_zeros),1) 
            L2 = torch.cat((L21,L22),1)
            L = torch.cat((L1.unsqueeze(1),L2.unsqueeze(1)),1)
        
        elif self.n == 3:
            L11 = y_L[:,0].unsqueeze(1)
            L1_zeros = torch.zeros(L11.size(0),2, dtype=torch.float64, device=device)

            L21 = y_L[:,1].unsqueeze(1)
            L22 = y_L[:,2].unsqueeze(1)
            L2_zeros = torch.zeros(L21.size(0),1, dtype=torch.float64, device=device)

            L31 = y_L[:,3].unsqueeze(1)
            L32 = y_L[:,4].unsqueeze(1)
            L33 = y_L[:,5].unsqueeze(1)

            L1 = torch.cat((L11,L1_zeros),1) 
            L2 = torch.cat((L21,L22,L2_zeros),1)
            L3 = torch.cat((L31,L32,L33),1)
            L = torch.cat((L1.unsqueeze(1),L2.unsqueeze(1),L3.unsqueeze(1)),1)
        
        elif self.n == 4:
            L11 = y_L[:,0].unsqueeze(1)
            L1_zeros = torch.zeros(L11.size(0),3, dtype=torch.float64, device=device)

            L21 = y_L[:,1].unsqueeze(1)
            L22 = y_L[:,2].unsqueeze(1)
            L2_zeros = torch.zeros(L21.size(0),2, dtype=torch.float64, device=device)

            L31 = y_L[:,3].unsqueeze(1)
            L32 = y_L[:,4].unsqueeze(1)
            L33 = y_L[:,5].unsqueeze(1)
            L3_zeros = torch.zeros(L31.size(0),1, dtype=torch.float64, device=device)

            L41 = y_L[:,6].unsqueeze(1)
            L42 = y_L[:,7].unsqueeze(1)
            L43 = y_L[:,8].unsqueeze(1)
            L44 = y_L[:,9].unsqueeze(1)

            L1 = torch.cat((L11,L1_zeros),1) 
            L2 = torch.cat((L21,L22,L2_zeros),1)
            L3 = torch.cat((L31,L32,L33,L3_zeros),1)
            L4 = torch.cat((L41,L42,L43,L44),1)
            L = torch.cat((L1.unsqueeze(1),L2.unsqueeze(1),L3.unsqueeze(1),L4.unsqueeze(1)),1)

        return L

    def get_A(self, a):
        if self.env_name == "pendulum" or self.env_name == "reacher":
            A = a
        
        elif self.env_name == "acrobot":
            A = torch.cat((self.a_zeros,a),1)
        
        elif self.env_name == "cartpole" or self.env_name == "cart2pole":
            A = torch.cat((a,self.a_zeros),1)
        
        elif self.env_name == "cart3pole" or self.env_name == "acro3bot":
            A = torch.cat((a[:,:1],self.a_zeros,a[:,1:]),1)

        return A

    def get_L(self, q):
        trig_q = self.trig_transform_q(q)
        L = self.compute_L(trig_q)         
        return L.sum(0), L

    def get_V(self, q):
        trig_q = self.trig_transform_q(q)
        y1_V = F.softplus(self.fc1_V(trig_q))
        y2_V = F.softplus(self.fc2_V(y1_V))
        V = self.fc3_V(y2_V).squeeze()
        return V.sum()

    def get_acc(self, q, qdot, a):
        dL_dq, L = jacrev(self.get_L, has_aux=True)(q)
        term_1 = torch.einsum('blk,bijk->bijl', L, dL_dq.permute(2,3,0,1))
        dM_dq = term_1 + term_1.transpose(2,3)
        c = torch.einsum('bjik,bk,bj->bi', dM_dq, qdot, qdot) - 0.5 * torch.einsum('bikj,bk,bj->bi', dM_dq, qdot, qdot)        
        Minv = torch.cholesky_inverse(L)
        dV_dq = 0 if self.env_name == "reacher" else jacrev(self.get_V)(q)
        qddot = torch.matmul(Minv,(self.get_A(a)-c-dV_dq).unsqueeze(2)).squeeze(2)
        return qddot                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                           
    def derivs(self, s, a):
        q, qdot = s[:,:self.n], s[:,self.n:]
        qddot = self.get_acc(q, qdot, a)
        return torch.cat((qdot,qddot),dim=1)                                                                                                                                                               

    def rk2(self, s, a):                                                                                                                                                                                   
        alpha = 2.0/3.0 # Ralston's method                                                                                                                                                                 
        k1 = self.derivs(s, a)                                                                                                                                                                             
        k2 = self.derivs(s + alpha * self.dt * k1, a)                                                                                                                                                      
        s_1 = s + self.dt * ((1.0 - 1.0/(2.0*alpha))*k1 + (1.0/(2.0*alpha))*k2)                                                                                                                            
        return s_1

    def forward(self, o, a):
        s_1 = self.rk2(self.inverse_trig_transform_model(o), a)
        o_1 = torch.cat((self.trig_transform_q(s_1[:,:self.n]),s_1[:,self.n:]),1)
        
        o_a = torch.cat((o, a), 1)
        rew = self.reward_net(o_a)
        
        return o_1, rew

    def get_next_state(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(obs)
        acs = np.atleast_2d(acs)
        
        # Predict next state deterministically
        next_obs, rew = self(ptu.from_numpy(obs), ptu.from_numpy(acs))

        return ptu.to_numpy(next_obs), ptu.to_numpy(rew)
    
    def update(self, obs, acs, rew, next_obs, **kwargs):
        pred_next_obs, pred_rew = self(obs, acs)

        loss = F.mse_loss(pred_next_obs, next_obs) + F.mse_loss(pred_rew, rew)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Env Training Loss': ptu.to_numpy(loss)
        }

class MLPDeterministicEnv(nn.Module):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs):
        super().__init__(**kwargs)

        # Init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # MLP for predicting next state deterministically
        backbone_layers = [
            nn.Linear(self.ac_dim + self.ob_dim, self.size), nn.ReLU(),
            *[layer for _ in range(self.n_layers - 1) for layer in (
                nn.Linear(self.size, self.size), nn.ReLU())],
        ]
        self.backbone_net = nn.Sequential(
            *backbone_layers,
        )
        self.state_pred_head = nn.Linear(self.size, self.ob_dim)
        self.reward_head = nn.Linear(self.size, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs, acs: torch.FloatTensor):
        """
        Forward pass of the network.

        Args:
            obs_acs (torch.FloatTensor): Input tensor containing observations and actions.

        Returns:
            torch.FloatTensor: Predicted next state.
        """
        input_tensor = torch.cat([obs, acs], dim=1)
        latent = self.backbone_net(input_tensor)
        next_obs = self.state_pred_head(latent)
        reward = self.reward_head(latent)
        
        return next_obs, reward

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def get_next_state(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(obs)
        acs = np.atleast_2d(acs)
        
        # Predict next state deterministically
        next_obs, rew = self(ptu.from_numpy(obs), ptu.from_numpy(acs))

        return ptu.to_numpy(next_obs), ptu.to_numpy(rew)

    def update(self, obs, acs, rew, next_obs, **kwargs):
        input_tensor = torch.cat([obs, acs], dim=1)

        pred_next_obs, pred_rew = self(obs, acs)

        loss = F.l1_loss(pred_next_obs, next_obs) + F.l1_loss(pred_rew, rew)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Env Training Loss': ptu.to_numpy(loss)
        }


class MLPStochasticEnv(nn.Module):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
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
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # Input: action and observation
        # Output: next observation (Note: reward is calculated separately)
        #self.logits_na = None
        self.mean_net = ptu.build_mlp(
            input_size=self.ac_dim+self.ob_dim,
            output_size=self.ob_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(
            torch.zeros(self.ob_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_next_state(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        """
          Single prediction of next state given observation and action.

        Args:
            obs (np.ndarray): _description_
            acs (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        # Ensure batch dimension
        obs = np.atleast_2d(obs)
        acs = np.atleast_2d(acs)
        
        # Merge observation and action
        obs_acs= np.concatenate([obs, acs], axis=1).astype(np.float32)
        obs_acs_tensor = ptu.from_numpy(obs_acs)
        # Get next obs
        next_obs_distrib = self.forward(obs_acs_tensor)
        next_obs = next_obs_distrib.sample() # Either batch or single value depends on input
        return ptu.to_numpy(next_obs)

    # update/train this policy
    def update(self, obs, acs, next_obs, **kwargs):
        """
        Train the model to minimize negative log-likelihood of next_observations
        given (observations, actions).
        obs, acs could be either batch or single datas
        """
        # Merge obs and actions into a single input
        input_tensor = torch.cat([obs, acs], dim=1)

        # Get the predicted distribution over next obs
        pred_distribution = self.forward(input)

        # Compute log-likelihood loss
        log_prob = pred_distribution.log_prob(next_obs)
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=1)
        loss = -log_prob.mean()
        
        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Env Training Loss': ptu.to_numpy(loss)
        }

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, ob_ac: torch.FloatTensor) -> Distribution:
        std = torch.exp(self.logstd)
        return torch.distributions.Normal(self.mean_net(ob_ac), std)





# ==========================================================

class BaseEnvModel(metaclass=abc.ABCMeta):
    """
    Base class for environment models in reinforcement learning.
    This class defines the interface for environment models, including methods
    for predicting the next state and reward given a current state and action.
    """

    def __init__(self):
        """
        Initialize the base environment model.
        """
        pass

    def step(self, state, action):
        """
        Predict the next state and reward given the current state and action.

        Args:
            state: The current state of the environment.
            action: The action taken by the agent.

        Returns:
            next_state: The predicted next state.
            reward: The predicted reward.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
  
    def train(self, states, actions, rewards, next_states):
        """
        Train the environment model using the provided data.

        Args:
            states: The current states of the environment.
            actions: The actions taken by the agent.
            rewards: The rewards received.
            next_states: The predicted next states.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
