from core.agents.agent import Agent
import random


class DummyAgent(Agent):
    def __init__(self, state_size, action_size, seed):
        super(DummyAgent, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state):
        return random.choice(range(self.action_size))

    def learn(self, experiences):
        pass
