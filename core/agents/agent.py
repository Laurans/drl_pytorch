class Agent:
    def __init__(self, agent_name, agent_params):
        # logging
        self.logger = agent_params.logger
        self.logger.info(
            "-----------------------------[ {} ]------------------".format(agent_name)
        )

        self.model_params = agent_params.model_params
        self.memory_params = agent_params.memory_params

        self.device = agent_params.device
        self.use_cuda = agent_params.use_cuda

        self.training = agent_params.training

        # hyperparameters
        self.steps = agent_params.steps
        self.gamma = agent_params.gamma
        self.clip_grad = agent_params.clip_grad
        self.lr = agent_params.lr
        self.eval_freq = agent_params.eval_freq
        self.eval_steps = agent_params.eval_steps
        self.test_nepisodes = agent_params.test_nepisodes

        self.learn_start = agent_params.learn_start
        self.batch_size = agent_params.batch_size
        self.valid_size = agent_params.valid_size
        self.eps = agent_params.eps_start
        self.eps_end = agent_params.eps_end
        self.eps_decay = agent_params.eps_decay
        self.eps_eval = agent_params.eps_eval
        self.target_model_update = agent_params.target_model_update
        self.action_repetition = agent_params.action_repetition
        self.optim = agent_params.optim
        self.tau = agent_params.tau
        self.learn_every = agent_params.learn_every

        self.counter_steps = 0

    def step(self, state, action, reward, next_state, done):
        raise NotImplementedError("not implemented step function in your agent")

    def act(self, state):
        raise NotImplementedError("not implemented act function in your agent")

    def learn(self, experiences):
        raise NotImplementedError("not implemented learn function in your agent")
