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
    
    def run_training_loop(self, rl_agent, env, test_env, env_model, replay_buffer_truth, replay_buffer_sim, hyperparams):
        raise NotImplementedError


class MBRL_Trainer(MBRL_TrainerBase): # Twin Delayed DDPG
    def __init__(self, base_params):
        super(MBRL_Trainer, self).__init__(base_params)
        self.debug_verbose = 0
        

    def run_training_loop(self, rl_agent: BaseAgent, env: Env, test_env: Env, env_model, replay_buffer_truth, replay_buffer_sim, hyperparams):
        # 0. Handle hyperparams init
        # Set random seeds
        if 'seed' in hyperparams:
            seed = hyperparams['seed']
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.add_noise = hyperparams['add_noise']
        self.start_steps = hyperparams['start_steps']
        self.noise_decay = hyperparams['noise_decay']
        self.update_after = hyperparams['update_after']
        self.update_every = hyperparams['update_every']
        self.batch_size = hyperparams['batch_size']
        
        self.action_max = env.action_space.high[0] # TODO: handle multi-dimensional actions
        self.action_min = env.action_space.low[0] # TODO: handle multi-dimensional actions
        
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
                if step_count < self.start_steps: # Encourage exploration in the beginning
                    ac = env.action_space.sample()
                else:
                    with torch.no_grad():
                        if self.add_noise:
                            ac_tensor = (
                                rl_agent.actor(ptu.from_numpy(obs_))
                                + torch.normal(0, self.action_max * noise_std_dev, size=(env.action_space.shape[0],))
                            ).clamp(self.action_min, self.action_max)
                            
                            if step_count % hyperparams['noise_decay_interval'] == 0:  # Decay every 5000 steps
                                noise_std_dev *= self.noise_decay
                        else:
                            ac_tensor = rl_agent.actor(ptu.from_numpy(obs_))
                            ac_tensor = ac_tensor.clamp(self.action_min, self.action_max)
                        ac = ptu.to_numpy(ac_tensor)

                # 2. Execute action and get next observation
                next_obs, rew, terminated, truncated, _ = env.step(ac)
                done = terminated or truncated
                # print shape of all variables
                #print(f"obs: {ptu.from_numpy(obs_).shape}, act: {ac_tensor.shape}, rew: {(rew)}, next_obs: {next_obs.shape}, done: {(done)}")
                # 3. Store (s, a, r, s', done) in replay buffer
                replay_buffer_truth.add(obs_, 
                                        ac, 
                                        rew, 
                                        next_obs, 
                                        1 if done else 0)

                if done:
                    obs_, _ = env.reset()
                else:
                    obs_ = next_obs
                    
                # =========== Model-based section ==============
                if step_count % hyperparams['learn_env_model_every'] == 0:
                    if step_count >= hyperparams['learn_env_model_after']:
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
                    if env_model_data_loss < hyperparams['env_model_loss_thresh']:
                        for _ in range(hyperparams['env_model_predict_steps']):
                            # Sample a batch of data from both replay buffer (ratio based on the size of each buffer)
                            _, _, _, obs_batch_tensor, _ = self._sample_mixed_batches(replay_buffer_truth, 
                                                                                      replay_buffer_sim, 
                                                                                      self.batch_size)
                            # Continue from the next_obs for one step using the environment model and curren policy
                            with torch.no_grad():
                                ac_batch_tensor = (
                                    rl_agent.actor(obs_batch_tensor) 
                                ).clamp(self.action_min, self.action_max)
                                # Predict next state and reward using the environment model
                                next_obs_pred_batch, rews_pred_batch = env_model.step(ptu.to_numpy(obs_batch_tensor), 
                                                                                     ptu.to_numpy(ac_batch_tensor))
                                if self.debug_verbose > 1:
                                    zipped_batch = zip(ptu.to_numpy(obs_batch_tensor), ptu.to_numpy(ac_batch_tensor), 
                                                       rews_pred_batch, next_obs_pred_batch)
                                    for obs, ac, rew_pred, next_obs_pred in zipped_batch:
                                        print(f"obs: {obs}, ac: {ac}, rew_pred: {rew_pred}, next_obs_pred: {next_obs_pred}")
                            # Store (s, a, r, s', done) in replay buffer
                            replay_buffer_sim.add(ptu.to_numpy(obs_batch_tensor), 
                                                ptu.to_numpy(ac_batch_tensor), 
                                                rews_pred_batch.reshape(-1,1), 
                                                next_obs_pred_batch, 
                                                np.zeros((self.batch_size,)).reshape(-1,1))
                # ============================================== 

                # 5. Determine if it's time to update
                #if len(self.replay_buffer.buffer) > self.batch_size:
                if step_count >= self.update_after and (step_count % self.update_every == 0):
                    # Perform n updates (currently set it the same number as the update_every)
                    for _ in range(self.update_every):
                        # 1. Sample a batch of data from replay buffer
                        obs_batch_tensor, ac_batch_tensor, rews_batch_tensor, next_obs_batch_tensor, done_batch_tensor \
                            = self._sample_mixed_batches(replay_buffer_truth, replay_buffer_sim, self.batch_size)
                        c_loss, a_loss = rl_agent.update(obs_batch_tensor, ac_batch_tensor, 
                                                         rews_batch_tensor, next_obs_batch_tensor, 
                                                         done_batch_tensor)
                    # TODO: sum loss?
                    print(f"c_loss: {c_loss}, a_loss: {a_loss}")

                    # Test/Evaluate the agent with test_env (run for 5 episodes)
                    self._evaluate_agent(rl_agent, test_env, num_episodes=5, step_idx=step_count)

                step_count += 1
            

    def _sample_mixed_batches(self, replay_buffer_truth, replay_buffer_sim, batch_size):
        total_size = replay_buffer_truth.size + replay_buffer_sim.size
        buffer_ratio = replay_buffer_truth.size / total_size if total_size > 0 else 1.0

        num_from_truth = round(batch_size * buffer_ratio)
        num_from_sim = batch_size - num_from_truth

        batch_obs, batch_ac, batch_rews, batch_next_obs, batch_done = replay_buffer_truth.sample(num_from_truth)

        if num_from_sim > 0 and replay_buffer_sim.size > 0:
            batch_obs_sim, batch_ac_sim, batch_rews_sim, batch_next_obs_sim, batch_done_sim = replay_buffer_sim.sample(num_from_sim)

            batch_obs = torch.cat((batch_obs, batch_obs_sim), dim=0)
            batch_ac = torch.cat((batch_ac, batch_ac_sim), dim=0)
            batch_rews = torch.cat((batch_rews, batch_rews_sim), dim=0)
            batch_next_obs = torch.cat((batch_next_obs, batch_next_obs_sim), dim=0)
            batch_done = torch.cat((batch_done, batch_done_sim), dim=0)

        return batch_obs, batch_ac, batch_rews, batch_next_obs, batch_done

    def _evaluate_agent(self, rl_agent, test_env, num_episodes=5, step_idx=0):
        total_rewards = []
        total_steps = 0

        for _ in range(num_episodes):
            obs, _ = test_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = rl_agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = test_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                obs = next_obs if not done else None
                total_steps += 1

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        avg_length = total_steps / num_episodes

        self.logger.log_reward(avg_reward, step_idx)
        print(f"Test episode reward avg: {avg_reward}, Test episode length avg: {avg_length}")
