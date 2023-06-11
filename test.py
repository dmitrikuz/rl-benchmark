import numpy as np
import gymnasium as gym

def last_mean(data):
    return round(np.mean(data[2:]), 5)


def get_data(env):

    return {
        'mean reward': last_mean(env.rewards),
        'mean path length': last_mean(env.path_lengths),
        'mean rotations': last_mean(env.rotations),
        'mean collisions': last_mean(env.collisions)
    }

def test_dqn(agent, env_name, epochs=30):
    env = gym.make(env_name, disable_env_checker=True, flat=False, old=True)
    agent.test(env, epochs, visualize=False)
    env.close()
    return get_data(env)


def test_usual(agent, env_name, epochs=30):
    env = gym.make(env_name, disable_env_checker=True, flat=True, old=False)
    for _ in range(epochs):

        done, truncated = False, False
        observation, _ = env.reset()

        while not (done or truncated):
            action = agent.get_best_action(observation)
            observation, reward, done, truncated, info = env.step(action)
        
    env.close()
    return get_data(env)