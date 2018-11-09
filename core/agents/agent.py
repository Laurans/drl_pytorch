class Agent:
    def __init__(self, agent_params):
        self.logger = agent_params.logger

    def step(self, state, action, reward, next_state, done):
        raise NotImplementedError("not implemented step function in your agent")

    def act(self, state):
        raise NotImplementedError("not implemented act function in your agent")

    def learn(self, experiences):
        raise NotImplementedError("not implemented learn function in your agent")
