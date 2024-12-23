import mj_envs
import gym
from stable_baselines3 import TD3

env = gym.make("door-v1")
model = TD3.load("model_best", env=env)


while True:
    obs = env.reset()
    env.env.mujoco_render_frames = True
    done = False
    while done is False:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        env.env.mj_render()