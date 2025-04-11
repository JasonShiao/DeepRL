import gymnasium as gym
from rl_physics.env_models.pendulum_env_model import PendulumEnvModel
from rl_physics.infrastructure.replay_buffer import ReplayBuffer
from rl_physics.infrastructure.rl_trainer import MBRL_Trainer
from rl_physics.infrastructure import pytorch_util as ptu

def run_training(usemodel, base_params):
    # Create trainer
    mbrl_trainer = MBRL_Trainer(base_params)

    # Make the gym environment
    env = gym.make(base_params['env_name'], render_mode = 'rgb_array')
    test_env = gym.make(base_params['env_name'], render_mode = 'rgb_array')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    # Create env model
    if base_params['env_name'] == 'Pendulum-v1':
        env_model = PendulumEnvModel(state_dim, action_dim)
    else:
        raise NotImplementedError("Env model not implemented for this environment")
    
    # Create Replay Buffers
    # Store data from real env
    replay_buffer_truth = ReplayBuffer(state_dim, action_dim, ptu.device, base_params['max_replay_buffer_size'])
    # Store data from real env + simulated from learned env model
    # Note: the simulated part will be reset once use
    replay_buffer_augmented = ReplayBuffer(state_dim, action_dim, ptu.device, base_params['max_replay_buffer_size']) 

    hyperparams =  {
        'add_noise': True,
        'warm_up_steps': 1000,
        'noise_decay': 1.0,
        'noise_decay_interval': 5000,
        'update_after': 500,
        'update_every': 50,
        'batch_size': 32,
        'seed': 0,
        'epoch': 20,
        'step_per_epoch': 1000,
        # Model-based related
        'learn_env_model_after': 100,
        'learn_env_model_every': 100,
        'env_model_loss_thresh': 0.05, # 0.01
        'env_model_predict_steps': 10,
        'usemodel': usemodel
    }
    agent_params = {
        'ac_dim': action_dim,
        'ob_dim': state_dim,
        'n_layers': 3,
        'size': 256,
        'discrete': False,
        'learning_rate': 2e-3,
        'policy_noise': 0.01,
        'noise_clip': 0.03,
        'max_action': float(env.action_space.high[0]), # TODO: handle multi-dimensional actions
        'min_action': float(env.action_space.low[0]), # TODO: handle multi-dimensional actions
        'gamma': 0.99,
        'polyak': 0.995,
        'policy_freq': 2,
    }

    # Create agent
    rl_agent = RLAgent(env, agent_params)
    
    # Train the agent
    mbrl_trainer.run_training_loop(
        rl_agent,
        env,
        test_env,
        env_model,
        replay_buffer_truth,
        replay_buffer_augmented,
        hyperparams = hyperparams
    )
    
    return rl_agent, mbrl_trainer


def plot_history(mbrl_logger, mfrl_logger):
    # plot reward log
    import matplotlib.pyplot as plt

    # Assuming logger is an instance of your Logger class
    plt.figure(figsize=(10, 5))
    mbrl_reward_mean, mbrl_reward_std, mbrl_reward_steps = mbrl_logger.get_reward_history()
    mfrl_reward_mean, mfrl_reward_std, mfrl_reward_steps = mfrl_logger.get_reward_history()
    plt.plot(mbrl_logger.reward_steps, mbrl_logger.reward_history, label='Avg MBRL reward')
    plt.fill_between(mbrl_reward_steps, 
                     mbrl_reward_mean - mbrl_reward_std,
                     mbrl_reward_mean + mbrl_reward_std, alpha=0.2)
    plt.plot(mfrl_logger.reward_steps, mfrl_logger.reward_history, label='Avg MFRL reward')
    plt.fill_between(mfrl_reward_steps, 
                     mfrl_reward_mean - mfrl_reward_std,
                     mfrl_reward_mean + mfrl_reward_std, alpha=0.2)
    plt.xlabel('Update step')
    plt.ylabel('Reward')
    plt.title('Reward History')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 5))
    mbrl_actor_mean, mbrl_actor_std, mbrl_actor_steps = mbrl_logger.get_actor_loss_history()
    mfrl_actor_mean, mfrl_actor_std, mfrl_actor_steps = mfrl_logger.get_actor_loss_history()
    plt.plot(mbrl_actor_steps, mbrl_actor_mean, label='MBRL Actor Loss')
    plt.fill_between(mbrl_actor_steps,
                    mbrl_actor_mean - mbrl_actor_std,
                    mbrl_actor_mean + mbrl_actor_std, alpha=0.2)
    plt.plot(mfrl_actor_steps, mfrl_actor_mean, label='MFRL Actor Loss')
    plt.fill_between(mfrl_actor_steps,
                    mfrl_actor_mean - mfrl_actor_std,
                    mfrl_actor_mean + mfrl_actor_std, alpha=0.2)
    plt.xlabel('Update step')
    plt.ylabel('Loss')
    plt.title('Actor Loss History')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 5))
    mbrl_critic_mean, mbrl_critic_std, mbrl_critic_steps = mbrl_logger.get_critic_loss_history()
    mfrl_critic_mean, mfrl_critic_std, mfrl_critic_steps = mfrl_logger.get_critic_loss_history()
    plt.plot(mbrl_critic_steps, mbrl_critic_mean, label='MBRL Critic Loss')
    plt.fill_between(mbrl_logger.actor_critic_loss_steps,
                    mbrl_critic_mean - mbrl_critic_std,
                    mbrl_critic_mean + mbrl_critic_std, alpha=0.2)
    plt.plot(mfrl_critic_steps, mfrl_critic_mean, label='MFRL Critic Loss')
    plt.fill_between(mfrl_logger.actor_critic_loss_steps,
                    mfrl_critic_mean - mfrl_critic_std,
                    mfrl_critic_mean + mfrl_critic_std, alpha=0.2)
    plt.xlabel('Update step')
    plt.ylabel('Loss')
    plt.title('Critic Loss History')
    plt.legend()
    plt.grid(True)
    plt.show()
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', '-alg', type=str, required=True) #relative to where you're running this script from
    #parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v4, Humanoid-v4, Walker-v4, HalfCheetah-v4, Hopper-v4', required=True)
    
    #parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    #parser.add_argument('--n_iter', '-n', type=int, default=1)
    
    #parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    #parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    #parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    #parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning
    
    #parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    #parser.add_argument('--which_gpu', type=int, default=0)
    #parser.add_argument('--save_params', action='store_true')
    #parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    
    if args.algo == 'ddpg':
        print("Using DDPG algorithm")
        from rl_physics.agents.ddpg_agent import DDPGAgent as RLAgent
    elif args.algo == 'td3':
        print("Using TD3 algorithm")
        from rl_physics.agents.td3_agent import TD3Agent as RLAgent
    else:
        raise NotImplementedError("Algorithm not implemented")

    base_params = {
      'env_name': 'Pendulum-v1',
      'logdir': 'logs',
      'no_gpu': False,
      'which_gpu': 0,
      'max_replay_buffer_size': 1000000,
    }

    mfrl_agent, mfrl_trainer = run_training(usemodel=False, base_params=base_params)
    #mbrl_agent, mbrl_trainer = run_training(usemodel=True, base_params=base_params)
    
    #plot_history(mbrl_trainer.logger, mfrl_trainer.logger)
    
    # Evaluate the agent
    eval_episodes = 10
    eval_env = gym.make(base_params['env_name'], render_mode = 'human')
    eval_env.reset()
    eval_env.render()
    avg_reward = 0.0
    for episode in range(eval_episodes):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action = mfrl_agent.get_action(ptu.from_numpy(obs), tensor=False)
            action = action.clip(-2, 2)  # Clip action to valid range
            obs, reward, terminated, truncated, info = eval_env.step(action)
            print(f"Action: {action}, Reward: {reward}, Next Obs: {obs}")
            done = terminated or truncated
            
            avg_reward += reward
            eval_env.render()
    
    #avg_reward /= eval_episodes
    #print(f"Average reward over {eval_episodes} episodes: {avg_reward}")
      
    