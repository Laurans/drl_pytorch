from core.agents.agent import Agent
import random


class DummyAgent(Agent):
    def __init__(self, agent_params, state_size, action_size):
        super(DummyAgent, self).__init__(agent_params)
        self.logger.info("-----------------------------[ Dummy ]------------------")

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(agent_params.seed)

    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state):
        return random.choice(range(self.action_size))

    def learn(self, experiences):
        pass
