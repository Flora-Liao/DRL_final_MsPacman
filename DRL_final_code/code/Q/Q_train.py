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



class Env():
    def __init__(self):
        self.env = gym.make("ALE/MsPacman-v5")
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

total_episodes = 500000
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

for episode in range(total_episodes):
    # Reset the environment
    print(f"Episode {episode + 1} Start")
    state, info = env.reset()
    state = preprocess_state(state)
    state_idx = hash_state(state, state_size)
    total_reward = 0
    done = False
    step = 0

    while not done:
        step += 1
        # 90% of the time, take the action that has the maximum Q-value
        if np.random.random() > epsilon:
            #print(Q_table[state_idx])
            action = torch.argmax(Q_table[state_idx]).item()
        # 10% of the time, take a random action
        else:
            action = np.random.randint(0, action_size)


        #print(action)
        new_state, reward, done, info = env.step(action)
        new_state = preprocess_state(new_state)
        new_state_idx = hash_state(new_state, state_size)
        reward = torch.tensor(reward, device=device)

        Q_table[state_idx, action] = Q_table[state_idx, action] + lr * (reward + gamma * torch.max(Q_table[new_state_idx]) - Q_table[state_idx, action])

        state = new_state
        state_idx = new_state_idx
        total_reward += reward.item()

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    episode_rewards.append(total_reward)
    print(f'Episode: {episode+1}, Total reward: {total_reward}')
    #print(Q_table)
    if (episode + 1) % 1000 == 0:
        # Save Q-table checkpoint
        torch.save(Q_table, f'/content/drive/MyDrive/DRL_final/Q/data/ckpt_episode_{episode+1}.pt')

        # Calculate 10-episode average reward
        if len(episode_rewards) >= 10:
            avg_reward = np.mean(episode_rewards[-10:])
        else:
            avg_reward = np.mean(episode_rewards)
        average_rewards.append(avg_reward)

        # Plot and save the 10-episode average reward graph
        plt.plot(average_rewards)
        plt.title("10-Episode Average Reward")
        plt.xlabel("Episode (x100)")
        plt.ylabel("Average Reward")
        plt.savefig(f'/content/drive/MyDrive/DRL_final/Q/pic/total_reward_episode_{episode+1}.png')
        plt.close()
        print(f'Checkpoint and plot saved for episode {episode + 1}')
