import os
import numpy as np
import torch
import gym
from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor

import panda_gym
from callback import SaveOnBestTrainingRewardCallback


def train(env_id, algo_class=DDPG, log_base_dir="logs", model_base_dir="models"):
    # set arguments --------------------------------------------------------------------------
    seed = 256
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)
    
    model_name = algo_class.__name__ + "_" + env_id
    model_path = os.path.join(model_base_dir, model_name)
    callback_log_path = os.path.join(model_base_dir, model_name, "temp")
    log_path = os.path.join(log_base_dir, model_name)

    # create the environment ------------------------------------------------------------------
    env = gym.make(env_id, render=False)
    env = Monitor(env, os.path.join(callback_log_path, 'monitor.csv'))

    # set the model : https://github.com/DLR-RM/rl-trained-agents/tree/master/her --------------
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
        online_sampling=True,
        max_episode_length=env.spec.max_episode_steps, 
    )

    policy_kwargs = {
        'net_arch': [512, 512, 512],
        'n_critics': 2,        
    }

    # if env_id == "PandaReachDense-v2":
    #     total_iterations = 50000
    #     learning_starts = 1000
    #     batch_size = 1024
    #     policy_kwargs = {
    #         'net_arch': [64, 64],
    #         'n_critics': 1,        
    #     }
    
    if env_id == "PandaReach-v2":
        total_iterations = 50_000
        learning_starts = 1000
        batch_size = 1024
    else:
        total_iterations = 3_000_000
        learning_starts = 1000
        batch_size = 2048
       
    model = algo_class(
        policy="MultiInputPolicy", 
        env=env, 
        buffer_size=1000000,
        learning_starts=learning_starts,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        gamma=0.95,
        # --------------------------------        
        tau=0.005,        
        learning_rate=0.001,
        train_freq=(1, "episode"),
        gradient_steps=1,    
        action_noise=action_noise, 
        optimize_memory_usage=False,
        # --------------------------------                   
        replay_buffer_class=HerReplayBuffer, 
        replay_buffer_kwargs=replay_buffer_kwargs,
        tensorboard_log=log_path,
        verbose=1,
        seed=seed,
        device='auto',        
    )

    # train the model -----------------------------------------------------------------------------
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=callback_log_path)
    model.learn(
        total_timesteps=total_iterations,
        callback=callback,
        log_interval=10,
        tb_log_name="DDPG",
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # save the trained model ----------------------------------------------------------------------
    model.save(model_path)
    print("model saved to {}".format(model_path))

    # close the environment
    del model
    env.close()

    
if __name__ == "__main__":
    env_id = "PandaReach-v2"    
    train(env_id)