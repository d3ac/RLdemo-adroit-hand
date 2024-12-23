import mj_envs
import gym
from stable_baselines3 import TD3

env = gym.make("door-v1")
model = TD3("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000, progress_bar=True, save_process=True, log_interval=1)

model.save("a2c_door_v1")
del model