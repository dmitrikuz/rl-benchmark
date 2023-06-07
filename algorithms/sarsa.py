from .qlearning import QLearningAgent


class SarsaAgent(QLearningAgent):
    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha

        current_q_value = self.get_qvalue(state, action)

        next_action = self.get_action(next_state)

        next_q_value = reward + gamma*self.get_qvalue(next_state, next_action)

        new_q_value = current_q_value + learning_rate*(reward + gamma*next_q_value - current_q_value)

        self.set_qvalue(state, action, new_q_value)