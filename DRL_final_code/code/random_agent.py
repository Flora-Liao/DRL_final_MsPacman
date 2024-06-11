import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import os

class Env():
    def __init__(self):
        self.env = gym.make("ALE/MsPacman-v5", render_mode="human")
        self.env.reset()
        self.is_done = False
        self.frame = deque([], maxlen=4)
        self.observation_space = (12, 210, 160)  # 4 frames stacked, each with 3 color channels
        self.action_space = 9
    
    def reset(self):
        obs, info = self.env.reset()
        for _ in range(4):
            self.frame.append(obs)
        return self._get_observation(), info
    
    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.frame.append(obs)
        self.is_done = done
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        assert len(self.frame) == 4
        return np.concatenate(self.frame, axis=2)

env = Env()
state_space = (12, 210, 160)
action_space = 9

num_rounds = 10
final_reward = []

for round_num in range(num_rounds):
    state, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        env.env.render()
        action = random.choice(range(9))
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
    print(f"Round {round_num + 1}: Total Reward = {total_reward}")
    final_reward.append(total_reward)

r = 0
for i in range(len(final_reward)):
   r += final_reward[i]
print(f"Average Reward: {r/len(final_reward)}")
env.env.close()