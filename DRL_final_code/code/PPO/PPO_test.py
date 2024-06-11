import argparse
from collections import deque
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta, Categorical
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train a PPO agent for MsPacman-v5')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--test', action='store_true', help='whether to test the model')
parser.add_argument('--model-path', type=str, default='', metavar='P', help='path to the model file')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

transition = np.dtype([
    ('s', np.float64, (210, 160, 12)), 
    ('a', np.float64, (1,)), 
    ('a_logp', np.float64),
    ('r', np.float64), 
    ('s_', np.float64, (210, 160, 12))
])

class Env():
    def __init__(self):
        self.env = gym.make("ALE/MsPacman-v5", render_mode="human")
        self.env.reset()
        self.is_done = False
        self.frame = deque([], maxlen=4)
        self.observation_space = (12, 210, 160)  # 4 frames stacked, each with 3 color channels
        self.action_space = 9
        self.reward_threshold = 1000  # An arbitrary reward threshold for stopping training

    def reset(self):
        obs, info = self.env.reset()
        #print(obs.shape)
        for _ in range(4):
            self.frame.append(obs)
        return self._get_observation(), None

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.frame.append(obs)
        self.is_done = done
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        assert len(self.frame) == 4
        return np.concatenate(self.frame, axis=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(22528, 512),
            nn.ReLU()
        )
        self.v = nn.Linear(512, 1)
        self.alpha_head = nn.Sequential(nn.Linear(512, 9), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(512, 9), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.reshape(x.size(0), -1)  # Changed from view to reshape
        x = self.fc(x)
        v = self.v(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return (alpha, beta), v

class Agent():
    max_grad_norm = 0.5
    clip_param = 0.1
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self):
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0).permute(0, 3, 1, 2)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        continuous_action = dist.sample().cpu()
        
        discrete_action = continuous_action.argmax(dim=1).item()
        a_logp = dist.log_prob(continuous_action).sum(dim=1).cpu().numpy()
        
        return discrete_action, a_logp

    def save_param(self, n):
        torch.save(self.net.state_dict(), f'./data/mspacman_{n}.pkl')

    def load_param(self, path):
        self.net.load_state_dict(torch.load(path, map_location=device))



def plot_rewards(rewards, filename):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.savefig(filename)
    plt.close()

def test(agent, env, num_episodes=10):
    total_rewards = []
    for i in range(num_episodes):
        env.env.render()
        state, _ = env.reset()
        episode_reward = 0
        while True:
            action, _ = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
        print(f'Episode {i + 1}: Total Reward: {episode_reward}')
    avg_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    return total_rewards

if __name__ == "__main__":
    agent = Agent()
    env = Env()

    
    agent.load_param("./data/mspacman_249.pkl")
    test(agent, env)
