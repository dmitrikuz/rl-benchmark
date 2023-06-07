


import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import torch
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
import argparse
import matplotlib.pyplot as plt
from gymnasium import register
from torch.nn.utils import clip_grad_norm_
import os, sys
import numpy as np

sys.path.append('..')


from environments import envs

render = False

NUM_EPISODES = 2000
GAMMA = 0.99
LR = 0.0001
ENV_NAME = 'SimpleMazeEnv-v0'


env = gym.make(ENV_NAME, disable_env_checker=True, flat=True, old=True)

n_states = env.reset().shape[0]
n_actions = 3
hidden = 378

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_states, hidden)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(hidden, n_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=LR)
eps = 0.01


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def train_reinforce(env, episodes=NUM_EPISODES):
    for _ in range(episodes):
        state = env.reset()
        for t in range(10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            if done:
                break
        finish_episode()

    env.close()
    return env


def run_reinforce(env_name=ENV_NAME, episodes=NUM_EPISODES, test=False):
    env = gym.make(env_name, disable_env_checker=True, flat=True, old=True)
    for _ in range(episodes):
        state = env.reset()
        for t in range(10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            if done:
                break
        
        if not test:
            finish_episode()

    env.close()
    return env

# if __name__ == '__main__':
#     train_reinforce(ENV_NAME, NUM_EPISODES)
#     print(np.mean(test_reinforce(ENV_NAME, 30).rewards))