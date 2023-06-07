import os
import gymnasium as gym
from gymnasium import register
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import os, sys

sys.path.append('..')


from environments import envs


ENV_NAME = 'BasicMazeEnv-v0'


#Parameters
env = gym.make(ENV_NAME, disable_env_checker=True, flat=True, old=False)
obs, _ = env.reset()

state_space = obs.shape[0]
action_space = 3


#Hyperparameters
LR = 0.0001
GAMMA = 0.99
render = False
hidden=512
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_space, hidden)

        self.action_head = nn.Linear(hidden, 3)
        self.value_head = nn.Linear(hidden, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_prob, state_values

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=LR, eps=0.01)


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    return action.item()


def finish_episode():
    R = 0
    save_actions = model.saved_actions
    policy_loss = []
    value_loss = []
    rewards = []

    for r in model.rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob , value), r in zip(save_actions, rewards):
        reward = r - value.item()
        policy_loss.append(-log_prob * reward)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]


def run_ac(env_name, episodes, test=False):

    env = env_name
    if isinstance(env_name, str): 
        env=gym.make(env_name, disable_env_checker=True, old=True, flat=True)

    for _ in range(episodes):
        state = env.reset()
        for _ in count():
            action = select_action(state)
            state, reward, done,  info = env.step(action)
            model.rewards.append(reward)

            if done:
                break

        if not test:
            finish_episode()

    env.close()
    return env


# run_ac(ENV_NAME, 10000)
