from core.agents.agent import Agent
import random

class MLPAgent(Agent):
    def __init__(self, agent_params, state_size, action_size, model_prototype):
        super(MLPAgent, self).__init__('MLP Agent', agent_params)

        self.model_params.state_shape = state_size
        self.model_params.action_dim = action_size
        self.model = model_prototype(self.model_params)

    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state):
        return 0

    def learn(self, experiences):
        pass