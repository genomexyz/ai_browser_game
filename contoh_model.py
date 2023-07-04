import gym_snakegame
import gymnasium as gym
import numpy as np

env = gym.make('gym_snakegame/SnakeGame-v0', size=25, n_target=1, render_mode='human')
# env = gym.wrappers.RecordVideo(env, video_folder='./video_folder', episode_trigger=lambda x: x % 200 == 0)

obs, info = env.reset()
anchor_step = 0
for i in range(100000):
    action = env.action_space.sample()
    #print('cek obs action', np.shape(observation), action)
    obs, reward, terminated, _, info = env.step(action)
    if terminated:
        print('total duration step', i - anchor_step)
        anchor_step = i
        env.reset()
env.close()