import os
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import panda_gym
from callback import SaveOnBestTrainingRewardCallback


def train(env_id, algo_class=PPO, log_base_dir="logs", model_base_dir="models"):
    # set arguments --------------------------------------------------------------------------
    seed = 256
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)

    model_name = algo_class.__name__ + "_" + env_id
    model_path = os.path.join(model_base_dir, model_name)
    callback_log_path = os.path.join(model_base_dir, model_name, "temp")
    log_path = os.path.join(log_base_dir, model_name)


    # create the environment ------------------------------------------------------------------
    env = make_vec_env(
        env_id=env_id, 
        n_envs=8, 
        seed=0, 
        monitor_dir=callback_log_path
    )
    
    # set the model : https://github.com/DLR-RM/rl-trained-agents/tree/master/her -------------
    if env_id == "PandaReachDense-v2":
        total_iterations = 100_000
        n_steps = 512
        batch_size = 256
        n_epochs = 32
        policy_kwargs = {'net_arch': [128, 128]}
    elif env_id == "PandaReach-v2":
        total_iterations = 500_000
        n_steps = 2048
        batch_size = 512
        n_epochs = 128
        policy_kwargs = {'net_arch': [512, 512]}        
    else:
        pass
        
    model = algo_class(
        policy="MultiInputPolicy", 
        env=env, 
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs,
        gamma=0.95,
        # ----------------------------
        learning_rate=0.001,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        # ----------------------------
        tensorboard_log=log_path,
        verbose=1,
        seed=seed,
        device='auto',                
    )

    # train the model -----------------------------------------------------------------------------
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, 
        log_dir=callback_log_path
    )
    model.learn(
        total_timesteps=total_iterations,
        callback=callback,
        log_interval=4,
        tb_log_name="PPO",
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
    