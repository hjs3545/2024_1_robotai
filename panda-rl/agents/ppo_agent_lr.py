# import os
# import torch
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor
# import pandas as pd
# from tqdm import tqdm

# import sys
# sys.path.append('/home/wilterovago/work/panda-rl/environments/simple_env/envs')
# from panda_lr import PandaEnv

# def make_env(rank, log_dir, seed=0):
#     def _init():
#         env = PandaEnv(use_gui=False)
#         env = Monitor(env, log_dir)  # Monitor wrapper to log statistics
#         env.seed(seed + rank)
#         return env
#     return _init

# def train(env_class, algo_class=PPO, log_base_dir="logs", model_base_dir="models"):
#     # set arguments --------------------------------------------------------------------------
#     seed = 256
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     model_name = algo_class.__name__ + "_" + env_class.__name__
#     model_path = os.path.join(model_base_dir, model_name)
#     callback_log_path = os.path.join(model_base_dir, model_name, "temp")
#     log_path = os.path.join(log_base_dir, model_name)

#     os.makedirs(callback_log_path, exist_ok=True)

#     # create the environment ------------------------------------------------------------------
#     n_envs = 8
#     env_fns = [make_env(i, callback_log_path) for i in range(n_envs)]
#     env = DummyVecEnv(env_fns)

#     # set the model -----------------------------------------------------------------------------
#     total_iterations = 500_000
#     n_steps = 2048
#     batch_size = 512
#     n_epochs = 128
#     policy_kwargs = {'net_arch': [512, 512]}

#     model = algo_class(
#         policy="MlpPolicy",  # 관측 공간이 단일 입력이므로 MlpPolicy 사용
#         env=env,
#         n_steps=n_steps,
#         batch_size=batch_size,
#         n_epochs=n_epochs,
#         policy_kwargs=policy_kwargs,
#         gamma=0.95,
#         # ----------------------------
#         learning_rate=0.001,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         normalize_advantage=True,
#         ent_coef=0.0,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         use_sde=False,
#         sde_sample_freq=-1,
#         target_kl=None,
#         # ----------------------------
#         tensorboard_log=log_path,
#         verbose=1,
#         seed=seed,
#         device='auto',  
#     )

#     # train the model -----------------------------------------------------------------------------
#     try:
#         with tqdm(total=total_iterations) as pbar:
#             model.learn(
#                 total_timesteps=total_iterations,
#                 log_interval=4,
#                 tb_log_name="PPO",
#                 reset_num_timesteps=True,
#                 progress_bar=True,
#             )
#     except KeyboardInterrupt:
#         print("Training interrupted. Saving model...")
#     finally:
#         pbar.close()

#     # save the trained model ----------------------------------------------------------------------
#     model.save(model_path)
#     print("model saved to {}".format(model_path))

#     # close the environment
#     del model
#     env.close()

# if __name__ == "__main__":
#     train(PandaEnv)


import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv  # 병렬 환경을 위해 SubprocVecEnv 사용
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('/home/wilterovago/work/panda-rl/environments/simple_env/envs')
from panda_lr import PandaEnv

def make_env(rank, log_dir, seed=0):
    def _init():
        env = PandaEnv(use_gui=False)
        env = Monitor(env, log_dir)  # Monitor wrapper to log statistics
        env.seed(seed + rank)
        return env
    return _init

def train(env_class, algo_class=PPO, log_base_dir="logs", model_base_dir="models"):
    # set arguments --------------------------------------------------------------------------
    seed = 256
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_name = algo_class.__name__ + "_" + env_class.__name__
    model_base_path = os.path.join(model_base_dir, model_name)
    callback_log_path = os.path.join(model_base_dir, model_name, "temp")
    log_path = os.path.join(log_base_dir, model_name)

    os.makedirs(callback_log_path, exist_ok=True)
    os.makedirs(model_base_path, exist_ok=True)

    # create the environment ------------------------------------------------------------------
    n_envs = 4
    env_fns = [make_env(i, callback_log_path) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)  # SubprocVecEnv 사용하여 환경 병렬화

    # set the model -----------------------------------------------------------------------------
    total_iterations = 1000000
    n_steps = 4096  # n_steps 증가
    batch_size = 1024  # batch_size 증가
    n_epochs = 10  # n_epochs 감소
    policy_kwargs = {'net_arch': [256, 256]}  # 모델 단순화

    model = algo_class(
        policy="MlpPolicy",  # 관측 공간이 단일 입력이므로 MlpPolicy 사용
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
    try:
        with tqdm(total=total_iterations) as pbar:
            model.learn(
                total_timesteps=total_iterations,
                log_interval=4,
                tb_log_name="PPO",
                reset_num_timesteps=True,
                progress_bar=True,
            )
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
    finally:
        pbar.close()

    # save the trained model ----------------------------------------------------------------------
    model_index = len(os.listdir(model_base_path)) + 1
    model_path = os.path.join(model_base_path, f"{model_name}_{model_index}_JS.zip")
    model.save(model_path)
    print("model saved to {}".format(model_path))

    # close the environment
    del model
    env.close()

if __name__ == "__main__":
    train(PandaEnv)
