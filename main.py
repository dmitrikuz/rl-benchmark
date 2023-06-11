#!/usr/bin/python3


import argparse
import os

import warnings

warnings.filterwarnings('ignore') 

import gymnasium as gym
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import logging

logging.getLogger().setLevel(logging.CRITICAL)

import csv
import json
from test import get_data, test_dqn, test_usual
import math
import matplotlib.colors as mcolors
import tensorflow as tf
from gymnasium.envs.registration import register
from matplotlib import pyplot as plt
from rl.agents.ddpg import DDPGAgent
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (Activation, Concatenate, Conv1D, Conv2D,
                                     Dense, Flatten, Input, MaxPooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers.legacy import Adam

from algorithms.ac import run_ac
from algorithms.qlearning import QLearningAgent
from algorithms.reinforce import run_reinforce
from algorithms.sarsa import SarsaAgent
from environments import envs



os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
parser = argparse.ArgumentParser(prog='Reinforcement Learning algorithms comparison for pathfinding problem',
                                description=
                                'The program performs the agent learning process based on user selected parameters \
                                and outputs the plots containing information about obtained rewards and other metrics selected by user'
)

parser.add_argument('-f', '--filename', help='File which is plot saved at', default='plot.png')
parser.add_argument('-a', '--algorithms', help='List of compared algorithms', nargs='+', default='Q-learning',
                     choices=['Q-learning', 'SARSA', 'DQN', 'DDQN', 'D3QN', 'REINFORCE', 'Actor-Critic'])
parser.add_argument('-l', '--learning-rate', help='Learning rate of training. Has to be in (0, 1)', type=float, default=0.001)
parser.add_argument('-d', '--discount-factor', help='Discount factor of training. Has to be in (0, 1]', type=float, default=0.99)
parser.add_argument('-n', '--episodes', help='Number of episodes of training process', default=5000, type=int)
parser.add_argument('-e', '--environment', help='Environment index', default=1, type=int)
parser.add_argument('-m', '--metrics', help='Metrics to be saved and displayed on plot', nargs='+', default=['rewards', 'paths'])
parser.add_argument('-p', '--epsilon', help='Randomness parameter', default=0.05, type=float)


args = parser.parse_args()
print(args)

env_name = 'SimpleMazeEnv-v0'
suffix = '5x5'
if args.environment == 2:
    env_name = 'BasicMazeEnv-v0'
    suffix = '10x10'
elif args.environment == 3:
    env_name = 'ComplexMazeEnv-v0'
    suffix = '20x20'

ENV_NAME = env_name
NUM_EPOCHS = args.episodes
EPSILON = args.epsilon
NUM_STEPS = NUM_EPOCHS * 50
NUM_ACTIONS = 3
LR = args.learning_rate
SUFFIX = suffix
GAMMA = args.discount_factor

colors = 'bgrcm'


def do_plots(env_dicts, prefix='', epochs=NUM_EPOCHS):

    observations_num = 100
    x = np.arange(0, NUM_EPOCHS, NUM_EPOCHS//observations_num)


    splitted = lambda data: [np.mean(part) for part in np.array_split(data, observations_num)]

    plt.xlabel('Number of episodes')
    plt.ylabel(f'Average reward for {NUM_EPOCHS//observations_num}  last episodes')
    

    # rewards
    for env_dict, color in zip(env_dicts, colors):
        rewards_splitted = splitted(env_dict['env'].rewards)
        plt.plot(x, rewards_splitted, label=env_dict['name'], color=color)

    plt.legend()
    plt.savefig(f'plots/{prefix}reward{SUFFIX}.png')
    plt.cla()

    plt.xlabel('Number of episodes')
    plt.ylabel(f'Average path length for {NUM_EPOCHS//observations_num} last episodes')

    # paths
    min_value, max_value = 1000, -1000
    for env_dict, color in zip(env_dicts, colors):
        paths_splitted = splitted(env_dict['env'].path_lengths)
        plt.plot(x, paths_splitted, label=env_dict['name'], color=color)

    plt.legend()
    plt.savefig(f'plots/{prefix}path{SUFFIX}.png')

    plt.cla()


def create_model():

        model = Sequential()
        model.add(Input(shape=(5, 9, 9, 3)))
        model.add(Conv2D(16, kernel_size=2, strides=1))
        model.add(Activation('relu'))
        model.add(Conv2D(32, kernel_size=2, strides=1))
        model.add(Activation('relu'))
        model.add(Conv2D(32, kernel_size=2, strides=1))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('linear'))

        return model


def run():

    to_plot, to_write = [], []

    if 'Q-learning' in args.algorithms:
        print('Q-LEARNING')
        ql_env = gym.make(ENV_NAME, disable_env_checker=True, flat=True)
        qlearning_agent = QLearningAgent(name='Q-Learning', alpha=LR, epsilon=EPSILON, discount=GAMMA, possible_actions=range(3))
        qlearning_agent.train(ql_env, NUM_EPOCHS)
        ql_test = test_usual(qlearning_agent, ENV_NAME)
        ql_test.update({'name': 'Q-learning'})
        to_plot.append({'env': ql_env, 'name': 'Q-Learning'})
        to_write.append(ql_test)
        print(ql_test)

    if 'SARSA' in args.algorithms:
        print('SARSA')
        sarsa_env = gym.make(ENV_NAME, disable_env_checker=True, flat=True)
        sarsa_agent = QLearningAgent(name='SARSA', alpha=LR, epsilon=EPSILON, discount=GAMMA, possible_actions=range(3))
        sarsa_agent.train(sarsa_env, NUM_EPOCHS)
        sarsa_test = test_usual(sarsa_agent, ENV_NAME)
        sarsa_test.update({'name': 'SARSA'})
        to_plot.append({'env': sarsa_env, 'name': 'SARSA'})
        to_write.append(sarsa_test)
        print(sarsa_test)

    if 'Actor-Critic' in args.algorithms:
        print("Actor-Critic")
        ac_env = gym.make(ENV_NAME, disable_env_checker=True, old=True, flat=True, agent_view_size=9)
        ac_env = run_ac(ENV_NAME, NUM_EPOCHS)
        ac_test = get_data(run_ac(ENV_NAME, 10, test=True))
        ac_test.update({'name': 'Actor-Critic'})
        to_plot.append({'env': ac_env, 'name': 'Actor-Critic'})
        to_write.append(ac_test)
        print(ac_test)

    if 'REINFORCE' in args.algorithms:
        print('REINFORCE')
        reinforce_env = gym.make(ENV_NAME, disable_env_checker=True, old=True, flat=True, agent_view_size=9)
        reinforce_env = run_reinforce(ENV_NAME, NUM_EPOCHS)
        reinforce_test = get_data(run_reinforce(ENV_NAME, 10, True))
        reinforce_test.update({'name': 'REINFORCE'})
        to_plot.append({'env': reinforce_env, 'name': 'REINFORCE'})
        to_write.append(reinforce_test)
        print(reinforce_test)



    if 'DQN' in args.algorithms:
        print('DQN')
        dqn_env = gym.make(
            id=ENV_NAME,
            disable_env_checker=True,
            old=True, 
            flat=False
        )

        dqn_agent = DQNAgent(
            model=create_model(),
            memory=SequentialMemory(limit=30000, window_length=5),
            policy=EpsGreedyQPolicy(EPSILON),
            nb_actions=NUM_ACTIONS,
            nb_steps_warmup=1000,
            gamma=GAMMA,
            batch_size=32,
            target_model_update=0.01,
        ) 

        dqn_agent.compile(Adam(learning_rate=LR))
        dqn_agent.fit(dqn_env, nb_steps=NUM_STEPS, visualize=False, verbose=0)
        dqn_agent.save_weights(f'weights/dqn{SUFFIX}', overwrite=True)
        dqn_test = test_dqn(dqn_agent, ENV_NAME, 30)
        dqn_test.update({'name': 'DQN'})
        to_plot.append({'env': dqn_env, 'name': 'DQN'})
        to_write.append(dqn_test)
        print(dqn_test)

    if 'DDQN' in args.algorithms:
        print('DDQN')
        ddqn_env = gym.make(
            id=ENV_NAME,
            disable_env_checker=True,
            old=True, 
            flat=False
        )
        ddqn_agent = DQNAgent(
            model=create_model(),
            memory=SequentialMemory(limit=30000, window_length=5),
            policy=EpsGreedyQPolicy(EPSILON),
            nb_actions=NUM_ACTIONS,
            nb_steps_warmup=1000,
            gamma=GAMMA,
            enable_double_dqn=True,
            batch_size=32,
            target_model_update=0.01,
        )
        ddqn_agent.compile(Adam(learning_rate=LR))
        ddqn_agent.fit(ddqn_env, nb_steps=NUM_STEPS, visualize=False, verbose=0)
        ddqn_agent.save_weights(f'weights/ddqn{SUFFIX}', overwrite=True)
        ddqn_test = test_dqn(ddqn_agent, ENV_NAME, 30)
        ddqn_test.update({'name': 'DDQN'})
        to_plot.append({'env': ddqn_env, 'name': 'DDQN'})
        to_write.append(ddqn_test)
        print(ddqn_test)

    if 'D3QN' in args.algorithms:
        print('D3QN')
        d3qn_env = gym.make(
            id=ENV_NAME,
            disable_env_checker=True,
            old=True, 
            flat=False
        )
        d3qn_agent = DQNAgent(
            model=create_model(),
            memory=SequentialMemory(limit=30000, window_length=5),
            policy=EpsGreedyQPolicy(EPSILON),
            nb_actions=NUM_ACTIONS,
            nb_steps_warmup=1000,
            gamma=GAMMA,
            enable_dueling_network=True,
            batch_size=32,
            target_model_update=0.01,
        )
        d3qn_agent.compile(Adam(learning_rate=LR))
        d3qn_agent.fit(d3qn_env, nb_steps=NUM_STEPS, visualize=False, verbose=0)
        d3qn_agent.save_weights(f'weights/d3qn{SUFFIX}', overwrite=True)
        d3qn_test = test_dqn(d3qn_agent, ENV_NAME, 30)
        d3qn_test.update({'name': 'DQN'})
        to_plot.append({'env': d3qn_env, 'name': 'D3QN'})
        to_write.append(d3qn_test)
        print(d3qn_test)

    do_plots(to_plot)
    print(to_write)

    with open(f'data/results{SUFFIX}.csv', 'w', newline='') as csvfile:
        fieldnames = ['name', 'mean reward', 'mean path length', 'mean collisions', 'mean rotations']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for env in to_write:
            writer.writerow(env)

if __name__ == '__main__':
    run()