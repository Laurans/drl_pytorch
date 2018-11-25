from core.utils.params import AgentParams


class Agent:
    def __init__(self, agent_name: str, agent_params: AgentParams) -> None:
        # logging
        self.logger = agent_params.logger

        self.model_params = agent_params.model_params
        self.memory_params = agent_params.memory_params

        self.device = agent_params.device
        self.use_cuda = agent_params.use_cuda

        self.training = agent_params.training

        # hyperparameters
        self.gamma = agent_params.gamma
        self.clip_grad = agent_params.clip_grad
        self.lr = agent_params.lr

        self.learn_start = agent_params.learn_start
        self.batch_size = agent_params.batch_size
        self.eps = agent_params.eps_start
        self.eps_end = agent_params.eps_end
        self.eps_decay = agent_params.eps_decay
        self.target_model_update = agent_params.target_model_update
        self.optim = agent_params.optim
        self.tau = agent_params.tau
        self.learn_every = agent_params.learn_every

        self.counter_steps = 0

        self.seed = agent_params.seed

        self.agent_name = agent_name
        self.model_dir = agent_params.model_dir

        self.logger.info(
            f"-----------------------------[ {agent_name} w/ seed {self.seed} ]------------------"
        )

    def step(self, state, action, reward, next_state, done):
        raise NotImplementedError("not implemented step function in your agent")

    def act(self, state):
        raise NotImplementedError("not implemented act function in your agent")

    def learn(self, experiences):
        raise NotImplementedError("not implemented learn function in your agent")

    def save(self, checkpoint):
        raise NotImplementedError("not implemented save function in your agent")

    def load(self, checkpoint):
        raise NotImplementedError("not implemented load function in your agent")
