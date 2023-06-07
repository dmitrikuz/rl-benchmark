class BaseAgent:
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.data = {
            'name': self.name,
            'rewards': [],
            'steps_counts': [],
            'collisions': [],
            'rotations': [],
            'convergence_time': [],
            'convergence_epochs': []
        }

    def update(self, state, action, reward, next_stater):
        ...

    def get_action(self, state):
        ...
    
    def get_best_action(self, state):
        ...

    def train(self):
        ...

    