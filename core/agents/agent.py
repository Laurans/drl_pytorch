class Agent:
    def __init__(self, agent_name, agent_params):
        # logging
        self.logger = agent_params.logger
        self.logger.info("-----------------------------[ {} ]------------------".format(agent_name))
        
        self.model_params = agent_params.model_params

    def step(self, state, action, reward, next_state, done):
        raise NotImplementedError("not implemented step function in your agent")

    def act(self, state):
        raise NotImplementedError("not implemented act function in your agent")

    def learn(self, experiences):
        raise NotImplementedError("not implemented learn function in your agent")
