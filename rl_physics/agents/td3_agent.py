from rl_physics.policies.MLP_policy import MLPDeterministicPolicy
from rl_physics.value_functions.MLP_critic import MLPTwinCritic
from .base_agent import BaseAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
import rl_physics.infrastructure.pytorch_util as ptu
import copy

"""
  TD3 is a off-policy actor-critic algorithm for continuous action spaces.
  TD3 can only be used for environments with continuous action spaces.

"""


class TD3Agent(BaseAgent):
    def __init__(self, env, agent_params):
        super(TD3Agent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPDeterministicPolicy(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
        )
        self.target_actor = copy.deepcopy(self.actor)
        # critic
        self.critic = MLPTwinCritic(
            self.agent_params['ob_dim'],
            self.agent_params['ac_dim'],
            n_layers=self.agent_params['n_layers'],
            size=self.agent_params['size'],
            learning_rate=self.agent_params['learning_rate'],
        )
        self.target_critic = copy.deepcopy(self.critic)
                
        self.total_iters = 0

    def update(self, ob_no, ac_na, re_n, next_ob_no, terminal_n): 
        """
          Train with a batch of data.

        Args:
            ob_no (_type_): _description_
            ac_na (_type_): _description_
            re_n (_type_): _description_
            next_ob_no (_type_): _description_
            terminal_n (_type_, optional): _description_

        Returns:
            _type_: _description_
        """
        self.total_iters += 1
        # training a TD3 agent refers to updating its actor and critic using
        # the given observations and corresponding action labels
                
        with torch.no_grad():
            # 1. Get next action from policy (next state) and clip it
            noise = (
                torch.randn_like(ac_na) * self.agent_params['policy_noise']
            ).clamp(-self.agent_params['noise_clip'], self.agent_params['noise_clip'])
            
            next_ac_na = (
                self.target_actor(next_ob_no) + noise
            ).clamp(-self.agent_params['max_action'], self.agent_params['max_action'])
            
            # 2. Compute target Q-value from critic network with action from policy network
            target_q1, target_q2 = self.target_critic(next_ob_no, next_ac_na)
            target_q = torch.min(target_q1, target_q2)
            # y(s', r, d) = r + gamma * (1-d) * min(Q_tar(s', a'))
            target_q = re_n + self.agent_params['gamma'] * (1 - terminal_n) * target_q

        # 2. Update critic(s) (gradient descent)
        current_q1, current_q2 = self.critic(ob_no, ac_na)
        c_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic.optimizer.zero_grad()
        c_loss.backward()
        #nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic.optimizer.step()
        
        # 3. Update actor (gradient ascent) with a period
        if self.total_iters % self.agent_params['policy_freq'] == 0:
            # Update policy (actor) with one of the critics only
            self.actor.optimizer.zero_grad()
            a_loss = -self.critic.Q1(ob_no, self.actor(ob_no)).mean() # Gradient ascent
            a_loss.backward()
            # Stablize the training
            #nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor.optimizer.step()

            # 4. Update "target" networks with polyak averaging
            with torch.no_grad():
                for target_param, eval_param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(self.agent_params['polyak'] * target_param.data + (1 - self.agent_params['polyak']) * eval_param.data)
                for target_param, eval_param in zip(self.target_critic.parameters(), self.critic.parameters()):
                    target_param.data.copy_(self.agent_params['polyak'] * target_param.data + (1 - self.agent_params['polyak']) * eval_param.data)
        else:
            a_loss = None

        # Print loss
        #print(f"critic loss = {c_loss.item()}, actor loss = {a_loss.item()}")
        return c_loss.item(), a_loss.item() if a_loss is not None else None
    
    def get_action(self, obs):
        obs_tensor = ptu.from_numpy(obs)
        self.target_actor.eval()
        with torch.no_grad():
            action_tensor = self.target_actor(obs_tensor)
        self.target_actor.train()
        return ptu.to_numpy(action_tensor)
