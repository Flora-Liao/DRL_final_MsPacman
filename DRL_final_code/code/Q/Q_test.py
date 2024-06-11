import gymnasium as gym
from collections import deque
import numpy as np
import torch
import os
import cv2
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(0)
if use_cuda:
    torch.cuda.manual_seed(0)

# Create directories for saving checkpoints and reward plots
#os.makedirs('./A2C_data', exist_ok=True)
#os.makedirs('./A2C_picture', exist_ok=True)

class Env():
    def __init__(self):
        self.env = gym.make("ALE/MsPacman-v5", render_mode="human")
        self.env.reset()
        self.is_done = False
        self.frame = deque([], maxlen=4)
        self.observation_space = (4, 64, 64)  # 4 frames stacked, each resized to 64x64
        self.action_space = 9
    
    def reset(self):
        obs, info = self.env.reset()
        for _ in range(4):
            self.frame.append(self._preprocess_frame(obs))
        return self._get_observation(), info
    
    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.frame.append(self._preprocess_frame(obs))
        self.is_done = done
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        assert len(self.frame) == 4
        return np.stack(self.frame, axis=0)

    def _preprocess_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized_frame = cv2.resize(gray_frame, (64, 64))
        return resized_frame

def preprocess_state(state):
    return torch.tensor(state.flatten(), dtype=torch.float32, device=device)

def hash_state(state, state_size):
    return int(torch.sum(state).item()) % state_size

env = Env()
env.reset()

action_size = 9
state_size = 1000000
Q_table = torch.zeros((state_size, action_size), device=device)

total_episodes = 5000
total_test_episodes = 10
max_steps = 200

lr = 0.01
gamma = 0.99

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.001
decay_rate = 0.01

# List to store the total rewards per episode for plotting
episode_rewards = []
average_rewards = []


# Load pretrained model
pretrained_model_path = './data/ckpt_episode_8000.pt'
Q_table = torch.load(pretrained_model_path, map_location=device)
#print(Q_table)
#for _ in Q_table:
    #print(_)
# Test the pretrained model
test_rewards = []
for episode in range(total_test_episodes):
    state, info = env.reset()
    state = preprocess_state(state)
    state_idx = hash_state(state, state_size)
    total_reward = 0
    done = False
    step = 0

    while not done:
        env.env.render()
        step += 1
        action = torch.argmax(Q_table[state_idx]).item()
        #print(action)
        new_state, reward, done, info = env.step(action)
        new_state = preprocess_state(new_state)
        state_idx = hash_state(new_state, state_size)
        total_reward += reward
        state = new_state

    test_rewards.append(total_reward)
    print(f'Test Episode: {episode+1}, Total reward: {total_reward}')

# Print average reward over test episodes
average_test_reward = np.mean(test_rewards)
print(f'Average Test Reward over {total_test_episodes} episodes: {average_test_reward}')
print(Q_table)
