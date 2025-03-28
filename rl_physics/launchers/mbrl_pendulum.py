import gymnasium as gym
from rl_physics.env_models.pendulum_env_model import PendulumEnvModel
from rl_physics.infrastructure.replay_buffer import ReplayBuffer
from rl_physics.infrastructure.rl_trainer import MBRL_Trainer
from rl_physics.infrastructure import pytorch_util as ptu

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--algo', '-alg', type=str, required=True) #relative to where you're running this script from
    #parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v4, Humanoid-v4, Walker-v4, HalfCheetah-v4, Hopper-v4', required=True)
    #parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    #parser.add_argument('--do_dagger', action='store_true')
    #parser.add_argument('--ep_len', type=int, default=1000)
    #
    #parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    #parser.add_argument('--n_iter', '-n', type=int, default=1)
    #
    #parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    #parser.add_argument('--eval_batch_size', type=int,
    #                    default=1000)  # eval data collected (in the env) for logging metrics
    #parser.add_argument('--train_batch_size', type=int,
    #                    default=100)  # number of sampled data points to be used per gradient/train step
    #
    #parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    #parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    #parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning
    #
    #parser.add_argument('--video_log_freq', type=int, default=5)
    #parser.add_argument('--scalar_log_freq', type=int, default=1)
    #parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    #parser.add_argument('--which_gpu', type=int, default=0)
    #parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
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
    replay_buffer_truth = ReplayBuffer(state_dim, action_dim, ptu.device, base_params['max_replay_buffer_size']) # real env
    replay_buffer_sim = ReplayBuffer(state_dim, action_dim, ptu.device, base_params['max_replay_buffer_size']) # simulated from learned env model

    hyperparams =  {
        'add_noise': True,
        'start_steps': 500,
        'noise_decay': 1.0,
        'noise_decay_interval': 5000,
        'update_after': 250,
        'update_every': 50,
        'batch_size': 32,
        'seed': 0,
        'epoch': 20,
        'step_per_epoch': 2000,
        # Model-based related
        'learn_env_model_after': 4000,
        'learn_env_model_every': 20,
        'env_model_loss_thresh': 0.04, # 0.01
        'env_model_predict_steps': 10,
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
        'max_action': env.action_space.high[0], # TODO: handle multi-dimensional actions
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
        replay_buffer_sim,
        hyperparams = hyperparams
    )
    
    # plot reward log
    import matplotlib.pyplot as plt

    # Assuming logger is an instance of your Logger class
    rewards = mbrl_trainer.logger.reward_history

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Avg reward over step')
    plt.xlabel('Update step')
    plt.ylabel('Reward')
    plt.title('Reward History')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    # Evaluate the agent
    eval_episodes = 5
    eval_env = gym.make(base_params['env_name'], render_mode = 'human')
    eval_env.reset()
    eval_env.render()
    avg_reward = 0.0
    for episode in range(eval_episodes):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action = rl_agent.get_action(obs)
            action = action.clip(-2, 2)  # Clip action to valid range
            obs, reward, terminated, truncated, info = eval_env.step(action)
            print(f"Action: {action}, Reward: {reward}, Next Obs: {obs}")
            done = terminated or truncated
            
            avg_reward += reward
            eval_env.render()
    
    avg_reward /= eval_episodes
    print(f"Average reward over {eval_episodes} episodes: {avg_reward}")
      
    