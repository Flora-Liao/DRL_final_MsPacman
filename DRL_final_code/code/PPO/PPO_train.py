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
        self.env = gym.make("ALE/MsPacman-v5")
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
        #print(f'state shape: {state.shape}')
        state = torch.from_numpy(state).double().to(device).unsqueeze(0).permute(0, 3, 1, 2)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        continuous_action = dist.sample().cpu()
        #print(f'continuous_action: {continuous_action}')
        
        # Select the action with the highest probability
        discrete_action = continuous_action.argmax(dim=1).item()
        a_logp = dist.log_prob(continuous_action).sum(dim=1).cpu().numpy()
        #print(f'discrete_action: {discrete_action}')
        
        return discrete_action, a_logp



    def save_param(self, n):
        torch.save(self.net.state_dict(), f'./data/mspacman_{n}.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        return False

    def update(self):
        self.training_step += 1
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device).permute(0, 3, 1, 2)  # Permute to [batch_size, channels, height, width]
        a = torch.tensor(self.buffer['a'], dtype=torch.long).to(device).view(-1)  # Change to long for Categorical
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device).permute(0, 3, 1, 2)  # Permute to [batch_size, channels, height, width]
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        print(f's type: {type(s)} s.shape: {s.shape}')
        print(f'a type: {type(a)} a.shape: {a.shape}')
        print(f'r type: {type(r)} r.shape: {r.shape}')
        print(f's_ type: {type(s_)} s_.shape: {s_.shape}')
        print(f'old_a_logp type: {type(old_a_logp)} old_a_logp.shape: {old_a_logp.shape}')

        # Debugging shapes before computing target_v and adv
        print(f'Before no_grad block:')
        print(f's_: {s_.shape}')
        print(f'r: {r.shape}')

        with torch.no_grad():
            s_values = self.net(s_)
            print(f's_values: {s_values}')
            
            target_v = r + args.gamma * s_values[1]
            adv = target_v - self.net(s)[1]
            
            # Ensure these print statements are reached
            print(f'target_v: {target_v.shape}, adv: {adv.shape}')

        for epoch in range(self.ppo_epoch):
            print(f'Starting PPO epoch {epoch + 1}/{self.ppo_epoch}')
            for batch_num, index in enumerate(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False)):
                print(f'Processing batch {batch_num + 1}')
                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                continuous_action = dist.sample()
                
                # Calculate action probabilities and use Categorical distribution
                action_probs = torch.softmax(continuous_action, dim=-1)
                action_dist = Categorical(action_probs)
                a_logp = action_dist.log_prob(a[index]).view(-1, 1)
                
                ratio = torch.exp(a_logp - old_a_logp[index])
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f'Batch {batch_num + 1} processed, loss: {loss.item()}')




def plot_rewards(rewards, filename):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    agent = Agent()
    env = Env()
    training_records = []
    total_rewards = []
    running_score = 0
    state, _ = env.reset()
    for i_ep in range(5000):
        score = 0
        state, _ = env.reset()
        for t in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, _ = env.step(action)
            #env.env.render()
            #state.permute(2, 0, 1)
            #state_.permute(2, 0, 1)
            #print(f'state shape: {state.shape}')
            #print(f'state_ shape: {state_.shape}')
            #print(f'action: {action}')
            if agent.store((state, action, a_logp, reward, state_)):
                print('Updating...')
                agent.update()
            score += reward
            state = state_
            if done:
                break
        total_rewards.append(score)
        #running_score = running_score * 0.99 + score * 0.01
        if (i_ep + 1) % 1 == 0:
            print(f'Episode {i_ep + 1} Score: {score}')
        if (i_ep + 1) % 10 == 0:
            agent.save_param(i_ep)
        if (i_ep + 1) % 50 == 0:
            np.save(f'total_rewards_{i_ep + 1}.npy', np.array(total_rewards))
            plot_rewards(total_rewards, f'./picture/total_rewards_{i_ep + 1}.png')
        #if running_score > env.reward_threshold:
         #   print(f"Solved! Running score is now {running_score / (i_ep + 1)}!")
          #  break

    plot_rewards(total_rewards, 'total_rewards_final.png')
