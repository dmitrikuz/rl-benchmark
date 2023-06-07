from .base import BaseAgent
from collections import defaultdict
import numpy as np
from time import perf_counter


class QLearningAgent(BaseAgent):
    def __init__(self, name, alpha, epsilon, discount, possible_actions):

        super().__init__(name)

        self.possible_actions = possible_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        state = tuple(state)
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        state = tuple(state)
        self._qvalues[state][action] = value

    def get_value(self, state):
    
        value = max([self.get_qvalue(state, action) for action in self.possible_actions])

        return value

    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha

        current_q_value = self.get_qvalue(state, action)
        new_q_value = (1 - learning_rate)*current_q_value + learning_rate*(reward + gamma * self.get_value(next_state))

        self.set_qvalue(state, action, new_q_value)

    def get_best_action(self, state):

        data = [(self.get_qvalue(state, action), action) for action in self.possible_actions]
        data = sorted(data,key=lambda x:x[0], reverse=True)
        result = [s[1] for s in data if s[0] == data[0][0]]
        best_action = np.random.choice(result)

        return best_action

    def get_action(self, state):
        chosen_action = None
        epsilon = self.epsilon
        prob = np.random.uniform()

        if prob <= epsilon:
            chosen_action =  np.random.choice(self.possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action
    

    def _train_for_episode(self, env, max_steps=10**4):
        state, info = env.reset()
        # steps, total_reward = 0, 0
        # collision_num, rotation_num = 0, 0

        for _ in range(max_steps):
            
            action = self.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            self.update(state, action, reward, next_state)
            state = next_state
            # total_reward += reward

            # if info['is_rotation']: rotation_num += 1
            # elif info['is_collision']: collision_num += 1
            # else: steps += 1


            if terminated or truncated:
                break

        # return total_reward, steps, rotation_num, collision_num



    def train(self, env, num_epochs):
        # eps = 0.8
        # convergence_epochs = 0
        # converged = False


        # start = perf_counter()
        # end = perf_counter()

        for epoch in range(num_epochs):

            # r, s, r_num, c_num = self._train_for_episode(env)
            self._train_for_episode(env)

            # self.data['rewards'].append(r)
            # self.data['steps_counts'].append(s)
            # self.data['collisions'].append(c_num)
            # self.data['rotations'].append(r_num)

            # last_100_std = np.std(self.data['rewards'][-100:])
            # last_100_mean = np.mean(self.data['rewards'][-100:])

            # if  last_100_std**2 < eps and last_100_mean > 0 and not converged:
            #     end = perf_counter()
            #     convergence_epochs = epoch
            #     converged = True

            # print(epoch, r)
        


        # convergence_time = end - start
        # self.data['convergence_epochs'].append(convergence_epochs)
        # self.data['convergence_time'].append(convergence_time)


