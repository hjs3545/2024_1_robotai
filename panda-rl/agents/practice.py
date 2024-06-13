import gym
from stable_baselines3 import PPO
import panda_gym

# 저장된 모델 경로
model_path = "/home/wilterovago/work/panda-rl/agents/models/PPO_PandaReach-v2/temp/best_model.zip"

# 모델 불러오기
model = PPO.load(model_path)

# 환경 생성
env = gym.make("PandaReach-v2")

# 시뮬레이션 실행
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()

# 환경 종료
env.close()
