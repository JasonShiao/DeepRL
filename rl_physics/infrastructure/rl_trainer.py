from collections import OrderedDict
import numpy as np
import time

import gymnasium as gym
import torch

from gymnasium import Env
from rl_physics.infrastructure import pytorch_util as ptu
from rl_physics.infrastructure.logger import Logger
from rl_physics.agents.base_agent import BaseAgent
import pickle 

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
# MAX_VIDEO_LEN = 40  # we overwrite this in the code below


def make_env(env_name, render_mode=('rgb_array')):
    if env_name == 'Ant-v4':
        return gym.make(env_name, use_contact_forces = True, render_mode = render_mode)
    else:
        return gym.make(env_name, render_mode = render_mode)


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


def make_env(env_name, render_mode=('rgb_array')):
    if env_name == 'Ant-v4':
        return gym.make(env_name, use_contact_forces = True, render_mode = render_mode)
    else:
        return gym.make(env_name, render_mode = render_mode)


class MBRL_Trainer(MBRL_TrainerBase): # Twin Delayed DDPG
    def __init__(self, base_params):
        super(MBRL_Trainer, self).__init__(base_params)
        

    def run_training_loop(self, rl_agent: BaseAgent, env: Env, test_env: Env, env_model, replay_buffer_truth, replay_buffer_sim, hyperparams):
        # 0. Handle hyperparams
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
                            #act = rl_agent.actor(ptu.from_numpy(obs_)) + torch.normal(0, noise_std_dev, size=(1,))
                            #act = self.actor(torch.as_tensor(obs_, dtype=torch.float32)) + torch.normal(0, noise_std_dev, size=(1,))
                            #act = act.cpu().detach().numpy()
                            #act = np.clip(act, env.action_space.low, env.action_space.high)
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
                        batch_obs, batch_ac, batch_rews, batch_next_obs, batch_done = replay_buffer_truth.sample(self.batch_size)
                        # Update/Train the environment model
                        train_report = env_model.train(batch_obs, batch_ac, batch_rews, batch_next_obs)
                        #print(f"c_loss: {c_loss}, a_loss: {a_loss}")
                        print(f"train_report: {train_report}, {train_report['Env Training Loss']}")
                        if np.isinf(env_model_data_loss):
                            env_model_data_loss = train_report['Env Training Loss']
                        else:
                            env_model_data_loss = 0.95 * env_model_data_loss + 0.05 * train_report['Env Training Loss']
                    if env_model_data_loss < hyperparams['env_model_loss_thresh']:
                        for _ in range(hyperparams['env_model_predict_steps']):
                            # Sample a batch of data from both replay buffer
                            # Split batch size based on the size of each buffer
                            buffer_ratio = replay_buffer_truth.size / (replay_buffer_truth.size + replay_buffer_sim.size)
                            _, _, _, batch_obs, _ = replay_buffer_truth.sample(round(self.batch_size * buffer_ratio))
                            if self.batch_size > round(self.batch_size * buffer_ratio):
                                _, _, _, batch_obs_sim, _ = replay_buffer_sim.sample(self.batch_size - round(self.batch_size * buffer_ratio))
                                # merge the two batches (tensors)
                                batch_obs = torch.cat((batch_obs, batch_obs_sim), dim=0)
                            # Continue from the next_obs for one step using the environment model and curren policy
                            # Get action from policy
                            with torch.no_grad():
                                ac_batch_tensor = rl_agent.actor(batch_obs)
                                ac_batch_tensor = ac_batch_tensor.clamp(self.action_min, self.action_max)
                                ac_batch = ptu.to_numpy(ac_batch_tensor)
                                # Predict next state and reward using the environment model
                                next_obs_pred_batch, rew_pred_batch = env_model.step(ptu.to_numpy(batch_obs), ac_batch)
                                zipped_batch = zip(ptu.to_numpy(batch_obs), ac_batch, rew_pred_batch, next_obs_pred_batch)
                                #for obs, ac, rew_pred, next_obs_pred in zipped_batch:
                                #    print(f"obs: {obs}, ac: {ac}, rew_pred: {rew_pred}, next_obs_pred: {next_obs_pred}")
                            # Store (s, a, r, s', done) in replay buffer
                            replay_buffer_sim.add(ptu.to_numpy(batch_obs), 
                                                ac_batch, 
                                                rew_pred_batch.reshape(-1,1), 
                                                next_obs_pred_batch, 
                                                np.zeros((self.batch_size,)).reshape(-1,1))
                # ============================================== 

                # 5. Determine if it's time to update
                #if len(self.replay_buffer.buffer) > self.batch_size:
                if step_count >= self.update_after and (step_count % self.update_every == 0):
                    # Perform n updates
                    buffer_ratio = replay_buffer_truth.size / (replay_buffer_truth.size + replay_buffer_sim.size)
                    #buffer_ratio = 1.0
                    num_data_from_truth = round(self.batch_size * buffer_ratio)
                    num_data_from_sim = self.batch_size - num_data_from_truth
                    for _ in range(self.update_every): # TBD: n updates
                        # 1. Sample a batch of data from replay buffer
                        batch_obs, batch_ac, batch_rews, batch_next_obs, batch_done = replay_buffer_truth.sample(num_data_from_truth)
                        if num_data_from_sim > 0:
                            batch_obs_sim, batch_ac_sim, batch_rews_sim, batch_next_obs_sim, batch_done_sim = replay_buffer_sim.sample(num_data_from_sim)
                            # merge the two batches (tensors)
                            batch_obs = torch.cat((batch_obs, batch_obs_sim), dim=0)
                            batch_ac = torch.cat((batch_ac, batch_ac_sim), dim=0)
                            batch_rews = torch.cat((batch_rews, batch_rews_sim), dim=0)
                            batch_next_obs = torch.cat((batch_next_obs, batch_next_obs_sim), dim=0)
                            batch_done = torch.cat((batch_done, batch_done_sim), dim=0)
                        #print(f"batch_obs: {batch_obs.shape}, batch_ac: {batch_ac.shape}, batch_rews: {batch_rews.shape}, batch_next_obs: {batch_next_obs.shape}, batch_done: {batch_done.shape}")
                        c_loss, a_loss = rl_agent.update(batch_obs, batch_ac, batch_rews, batch_next_obs, batch_done)
                    print(f"c_loss: {c_loss}, a_loss: {a_loss}")

                    # Test/Evaluate the agent with test_env (run for 5 episodes)
                    test_step_count = 0
                    test_rews = []
                    for _ in range(5):
                        test_obs, _ = test_env.reset()
                        test_done = False
                        while not test_done:
                            test_ac = rl_agent.get_action(test_obs)
                            test_next_obs, test_rew, test_terminated, test_truncated, _ = test_env.step(test_ac)
                            test_rews.append(test_rew)
                            test_done = test_terminated or test_truncated
                            if not test_done:
                                test_obs = test_next_obs
                                test_step_count += 1
                    
                    test_rew_sum = np.sum(test_rews)
                    self.logger.log_reward(test_rew_sum / 5, step_count)
                    print(f"Test episode reward avg: {test_rew_sum / 5}, Test episode length: {test_step_count}")                    

                step_count += 1
            


