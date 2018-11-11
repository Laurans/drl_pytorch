import os
import visdom
import torch

from .logger import loggerConfig
from collections import namedtuple
import torch.optim as optim


class Params:
    def __init__(self, verbose):
        self.verbose = 1  # 0 (no set) | 1 (info) | 2 (debug)

        # signature
        self.machine = "machine"
        self.timestamp = "181109"

        #
        self.seed = 123
        self.visualize = False

        # prefix for saving
        self.refs = self.machine + "_" + self.timestamp
        self.root_dir = os.getcwd()

        # logging config
        self.log_name = self.root_dir + "/logs/" + self.refs + ".log"
        self.logger = loggerConfig(self.log_name, self.verbose)
        self.logger.info(":===================================:")

        if self.visualize:
            self.vis = visdom.Visdom()
            self.logger.info("bash$: python3 -m visdom.server")
            self.logger.info("http://localhost:8097/env/{}".format(self.refs))

        self.use_cuda = torch.cuda.is_available()
        self.dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")


class ModelParams(Params):
    def __init__(self, verbose):
        super(ModelParams, self).__init__(verbose)

        self.hist_len = 1
        self.hidden_dim = 64


class MemoryParams(Params):
    def __init__(self, verbose):
        super(MemoryParams, self).__init__(verbose)

        self.memory_size = int(1e5)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )


class AgentParams(Params):
    def __init__(self, verbose):
        super(AgentParams, self).__init__(verbose)

        self.model_params = ModelParams(verbose)
        self.memory_params = MemoryParams(verbose)

        self.training = True

        # hyperparameters
        self.steps = 100000
        self.gamma = 0.99
        self.clip_grad = 1.0
        self.lr = 5e-4
        self.eval_freq = 2500
        self.eval_steps = 1000
        self.test_nepisodes = 1

        self.learn_start = 500
        self.learn_every = 4
        self.batch_size = 64
        self.valid_size = 250
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.eps_eval = 0.0
        self.target_model_update = 1000
        self.action_repetition = 1

        self.optim = optim.Adam
        self.tau = 1e-3

        self.memory_params.window_length = self.model_params.hist_len - 1
