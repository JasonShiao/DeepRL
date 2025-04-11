from collections import OrderedDict
import numpy as np
import time

import gymnasium as gym
from gymnasium import Env
import torch

from rl_physics.infrastructure import pytorch_util as ptu
from rl_physics.infrastructure.logger import Logger
from rl_physics.agents.base_agent import BaseAgent
import pickle 

# how many rollouts to save as videos to tensorboard
#MAX_NVIDEO = 2
# MAX_VIDEO_LEN = 40  # we overwrite this in the code below

#def make_env(env_name, render_mode=('rgb_array')):
#    if env_name == 'Ant-v4':
#        return gym.make(env_name, use_contact_forces = True, render_mode = render_mode)
#    else:
#        return gym.make(env_name, render_mode = render_mode)

class MBRL_TrainerBase(object):
    def __init__(self, base_params):
        self.base_params = base_params
        self.logger = Logger(self.base_params['logdir'])

        ptu.init_gpu(
            use_gpu=not self.base_params['no_gpu'],
            gpu_id=self.base_params['which_gpu']
        )
    
    def run_training_loop(self, rl_agent, env, test_env, env_model, replay_buffer_truth, replay_buffer_augmented, hyperparams):
        raise NotImplementedError


class MBRL_Trainer(MBRL_TrainerBase): # Twin Delayed DDPG
    def __init__(self, base_params):
        super(MBRL_Trainer, self).__init__(base_params)
        self.debug_verbose = 0
        

    def run_training_loop(self, rl_agent: BaseAgent, env: Env, test_env: Env, env_model, replay_buffer_truth, replay_buffer_augmented, hyperparams):
        # 0. Handle hyperparams init
        # Set random seeds
        if 'seed' in hyperparams:
            seed = hyperparams['seed']
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.add_noise = hyperparams['add_noise']
        self.warm_up_steps = hyperparams['warm_up_steps']
        self.noise_decay = hyperparams['noise_decay']
        self.update_after = hyperparams['update_after']
        self.update_every = hyperparams['update_every']
        self.batch_size = hyperparams['batch_size']
        
        self.action_max = env.action_space.high[0] # TODO: handle multi-dimensional actions
        self.action_min = env.action_space.low[0] # TODO: handle multi-dimensional actions
        
        self.usemodel = hyperparams['usemodel'] # True # False
        
        # TODO: remove hard coded values
        noise_std_dev = 0.1 # (self.env.action_space.high - self.env.action_space.low) / 6
        if 'seed' in hyperparams:
            obs_, _ = env.reset(seed=seed)
        else:
            obs_, _ = env.reset()
        step_count = 0
        
        env_model_data_loss = np.inf
        
        for epoch_idx in range(hyperparams['epoch']):
            for _ in range(hyperparams['step_per_epoch']): # TBD: epoch?
                # 1. Get action (off-policy for the first n steps, then use policy)
                if step_count < self.warm_up_steps: # Encourage exploration in the beginning
                    ac = env.action_space.sample()
                else:
                    with torch.no_grad():
                        if self.add_noise:
                            ac_tensor = (
                                rl_agent.get_action(ptu.from_numpy(obs_), tensor=True)
                                + torch.normal(0, self.action_max * noise_std_dev, size=(env.action_space.shape[0],))
                            ).clamp(self.action_min, self.action_max)
                            
                            if step_count % hyperparams['noise_decay_interval'] == 0:  # Decay every 5000 steps
                                noise_std_dev *= self.noise_decay
                        else:
                            ac_tensor = rl_agent.get_action(ptu.from_numpy(obs_), tensor=True)
                        ac = ptu.to_numpy(ac_tensor)

                # 2. Execute action and get next observation
                next_obs, rew, terminated, truncated, _ = env.step(ac)
                done = terminated or truncated
                # print shape of all variables
                #print(f"obs: {ptu.from_numpy(obs_).shape}, act: {ac_tensor.shape}, rew: {(rew)}, next_obs: {next_obs.shape}, done: {(done)}")
                # 3. Store (s, a, r, s', done) to "both" replay buffers
                replay_buffer_truth.add(obs_, ac, rew, next_obs, 
                                        1 if done else 0)
                #replay_buffer_augmented.add(obs_, ac, rew, next_obs, 
                #                        1 if done else 0)

                if done:
                    obs_, _ = env.reset()
                else:
                    obs_ = next_obs
                    
                # =========== Model-based section ==============
                # 4. Train the environment model
                if step_count % hyperparams['learn_env_model_every'] == 0 and self.usemodel:
                    if step_count >= hyperparams['learn_env_model_after']:
                        max_model_iter = 250
                        for model_iter in range(max_model_iter):
                            # Sample a batch of data from truth replay buffer
                            obs_batch_tensor, ac_batch_tensor, rews_batch_tensor, \
                                next_obs_batch_tensor, done_batch_tensor = replay_buffer_truth.sample(self.batch_size)
                            # Update/Train the environment model
                            train_report = env_model.train(obs_batch_tensor, ac_batch_tensor, 
                                                        rews_batch_tensor, next_obs_batch_tensor)
                            if self.debug_verbose > 0:
                                print(f"train_report: {train_report}, {train_report['Env Training Loss']}")
                            if np.isinf(env_model_data_loss):
                                env_model_data_loss = train_report['Env Training Loss']
                            else:
                                env_model_data_loss = 0.95 * env_model_data_loss + 0.05 * train_report['Env Training Loss']
                            
                            if model_iter > 10 and env_model_data_loss < (hyperparams['env_model_loss_thresh'] / 10):
                                break

                        print(f"env_model_data_loss: {env_model_data_loss}, env_model_loss_thresh: {hyperparams['env_model_loss_thresh']}")
                # ============================================== 
                
                # 5. Determine if it's time to update agent with "real" replay buffer
                #if len(self.replay_buffer.buffer) > self.batch_size:
                if step_count >= self.update_after and (step_count % self.update_every == 0):
                    # Perform n updates (currently set it the same number as the update_every)
                    c_loss_list = []
                    a_loss_list = []
                    c_loss_avg = 0
                    a_loss_avg = 0
                    actor_update_count = 0
                    for _ in range(self.update_every):
                        # 1. Sample a batch of data from replay buffer
                        obs_batch_tensor, ac_batch_tensor, rews_batch_tensor, next_obs_batch_tensor, done_batch_tensor \
                            = replay_buffer_truth.sample(self.batch_size)
                        c_loss, a_loss = rl_agent.update(obs_batch_tensor, ac_batch_tensor, 
                                                         rews_batch_tensor, next_obs_batch_tensor, 
                                                         done_batch_tensor)
                        c_loss_list.append(c_loss)
                        c_loss_avg += c_loss
                        if a_loss is not None:
                            actor_update_count += 1
                            a_loss_avg += a_loss
                            a_loss_list.append(a_loss)

                    c_loss = c_loss_avg / self.update_every
                    a_loss = a_loss_avg / actor_update_count
                    c_loss_std = np.std(c_loss_list)
                    a_loss_std = np.std(a_loss_list) if actor_update_count > 0 else 0
                    self.logger.log_actor_critic_loss(a_loss, a_loss_std, c_loss, c_loss_std, step_count)
                    # TODO: sum loss?
                    #print(f"c_loss: {c_loss}, a_loss: {a_loss}")
                
                # =========== Model-based section ==============
                # 6. If model is accurate enough, use it to augment the augment replay buffer and train the agent
                if step_count % hyperparams['learn_env_model_every'] == 0 and step_count >= hyperparams['learn_env_model_after'] and self.usemodel:
                    #print(f"env_model_data_loss: {env_model_data_loss}, env_model_loss_thresh: {hyperparams['env_model_loss_thresh']}")
                    if env_model_data_loss < hyperparams['env_model_loss_thresh']:
                        # num interaction
                        fake_interact_count = 1000
                        _, _, _, obs_batch_tensor, _ = replay_buffer_truth.sample(self.batch_size)
                        for t in range(fake_interact_count):
                            # Sample a batch of data from real replay buffer
                            # Continue from the next_obs for one step using the environment model and curren policy
                            with torch.no_grad():
                                ac_batch_tensor = rl_agent.get_action(obs_batch_tensor, tensor=True)
                                # Predict next state and reward using the environment model
                                next_obs_pred_batch, rews_pred_batch = env_model.step(ptu.to_numpy(obs_batch_tensor), 
                                                                                     ptu.to_numpy(ac_batch_tensor))
                                if self.debug_verbose > 1:
                                    zipped_batch = zip(ptu.to_numpy(obs_batch_tensor), ptu.to_numpy(ac_batch_tensor), 
                                                       rews_pred_batch, next_obs_pred_batch)
                                    for obs, ac, rew_pred, next_obs_pred in zipped_batch:
                                        print(f"obs: {obs}, ac: {ac}, rew_pred: {rew_pred}, next_obs_pred: {next_obs_pred}")
                            # Store (s, a, r, s', done) in augmented replay buffer only
                            replay_buffer_augmented.add(ptu.to_numpy(obs_batch_tensor), 
                                                ptu.to_numpy(ac_batch_tensor), 
                                                rews_pred_batch.reshape(-1,1), 
                                                next_obs_pred_batch, 
                                                np.zeros((self.batch_size,)).reshape(-1,1))
                            if t % hyperparams['env_model_predict_steps'] == 0:
                                _, _, _, obs_batch_tensor, _ = replay_buffer_truth.sample(self.batch_size)

                        
                        # Train agent using the augmented replay buffer
                        c_loss_avg = 0
                        a_loss_avg = 0
                        actor_update_count = 0
                        for _ in range(20):
                            # 1. Sample a batch of data from replay buffer
                            obs_batch_tensor, ac_batch_tensor, rews_batch_tensor, next_obs_batch_tensor, done_batch_tensor \
                                = replay_buffer_augmented.sample(self.batch_size)
                            c_loss, a_loss = rl_agent.update(obs_batch_tensor, ac_batch_tensor, 
                                                            rews_batch_tensor, next_obs_batch_tensor, 
                                                            done_batch_tensor)
                            c_loss_avg += c_loss
                            if a_loss is not None:
                                actor_update_count += 1
                                a_loss_avg += a_loss

                        c_loss = c_loss_avg / self.update_every
                        a_loss = a_loss_avg / actor_update_count
                        #self.logger.log_actor_critic_loss(a_loss, c_loss, step_count)
                        # TODO: sum loss?
                        #print(f"c_loss: {c_loss}, a_loss: {a_loss}")
                        
                        # Immediatedly discard the simulated part from augmented replay buffer
                        replay_buffer_augmented.trim(0)
                # ==============================================
                
                
                # Test/Evaluate the agent with test_env (run for 5 episodes)
                # TODO: Check time to evaluate
                if step_count % 500 == 0:
                    self._evaluate_agent(rl_agent, test_env, num_episodes=5, step_idx=step_count)


                step_count += 1

    def _evaluate_agent(self, rl_agent, test_env, num_episodes=5, step_idx=0):
        total_rewards = []
        total_steps = 0

        for _ in range(num_episodes):
            obs, _ = test_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = rl_agent.get_action(ptu.from_numpy(obs), tensor=False)
                next_obs, reward, terminated, truncated, _ = test_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                obs = next_obs if not done else None
                total_steps += 1

            total_rewards.append(episode_reward)

        reward_avg = np.mean(total_rewards)
        reward_std = np.std(total_rewards)
        avg_length = total_steps / num_episodes

        self.logger.log_reward(reward_avg, reward_std, step_idx)
        print(f"Step_idx: {step_idx}. Test episode reward avg: {reward_avg}, Test episode length avg: {avg_length}")
