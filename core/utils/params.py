import os
import visdom
import torch
import imageio

from .logger import loggerConfig
from collections import namedtuple
import torch.optim as optim


class Params:
    def __init__(
        self,
        verbose: int,
        machine: str = "machine",
        timestamp: str = "",
        visualize: bool = False,
        env_render: bool = False,
    ) -> None:
    """Object params that contains all the common variables between modules like logger or GPU device
    
    Args:
        verbose (int): level of verbosity
        machine (str, optional): Defaults to "machine". Machine name where the algorithm is run. Used to create logging filename signature
        timestamp (str, optional): Defaults to "". Time where the algorithm is run. Used to create logging filename signature
        visualize (bool, optional): Defaults to False. Set connection to visdom dashboard if true
        env_render (bool, optional): Defaults to False. Save evaluation images in directory to used later
    
    """

        self.verbose = verbose  # 0 (no set) | 1 (info) | 2 (debug)

        # signature
        self.machine = machine
        self.timestamp = timestamp

        #
        self.seed = 123
        self.visualize = visualize
        self.env_render = env_render

        # prefix for saving
        self.refs = self.machine + "_" + self.timestamp
        self.root_dir = os.getcwd()

        # logging config
        self.log_name = self.root_dir + "/logs/" + self.refs + ".log"
        self.logger = loggerConfig(self.log_name, self.verbose)

        if self.visualize:
            self.vis = visdom.Visdom()
            self.logger.info("bash$: python3 -m visdom.server")
            self.logger.info("http://localhost:8097/env/{}".format(self.refs))

        self.use_cuda = torch.cuda.is_available()
        self.dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")


class ModelParams(Params):

    def __init__(
        self, verbose: int, machine: str = "machine", timestamp: str = ""
    ) -> None:
    """Model global parameters
    
    Args:
        verbose (int): Level of verbosity
        machine (str, optional): Defaults to "machine". Machine name where the algorithm is run. Used to create logging filename signature
        timestamp (str, optional): Defaults to "". Time where the algorithm is run. Used to create logging filename signature
    
    """


        super(ModelParams, self).__init__(verbose, machine=machine, timestamp=timestamp)

        self.hist_len = 1
        self.hidden_dim = 64

        self.state_shape = None
        self.action_dim = None


class MemoryParams(Params):

    def __init__(
        self, verbose: int, machine: str = "machine", timestamp: str = ""
    ) -> None:
    """Memory global parameters
    
    Args:
        verbose (int): Level of verbosity
        machine (str, optional): Defaults to "default". Machine name where the algorithm is run. Used to create logging filename signature
        timestamp (str, optional): Defaults to "". Time where the algorithm is run. Used to create logging filename signature
    
    """

        super(MemoryParams, self).__init__(
            verbose, machine=machine, timestamp=timestamp
        )

        self.memory_size = int(1e5)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

        self.window_length = 0


class AgentParams(Params):

    def __init__(
        self, verbose: int, machine: str = "default", timestamp: str = ""
    ) -> None:
    """Agent global parameters. It contains Model and Memory Parameters
    
    Args:
        verbose (int): Level of verbosity
        machine (str, optional): Defaults to "default". Machine name where the algorithm is run. Used to create logging filename signature
        timestamp (str, optional): Defaults to "". Time where the algorithm is run. Used to create logging filename signature
    """

        super(AgentParams, self).__init__(verbose, machine=machine, timestamp=timestamp)

        self.model_params = ModelParams(verbose, machine=machine, timestamp=timestamp)
        self.memory_params = MemoryParams(verbose, machine=machine, timestamp=timestamp)

        self.training = True

        # hyperparameters
        self.gamma = 0.99
        self.clip_grad = 1.0
        self.lr = 5e-4

        self.learn_start = 500
        self.learn_every = 4
        self.batch_size = 64

        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.target_model_update = 1000

        self.optim = optim.Adam
        self.tau = 1e-3
        self.update_every = 4

        self.memory_params.window_length = self.model_params.hist_len - 1


class MonitorParams(Params):

    def __init__(
        self,
        verbose: int,
        machine: str = "machine",
        timestamp: str = "",
        visualize: bool = False,
        env_render: bool = False,
    ):
    """Monitor global parameters. It contains an AgentParams object and set visualisation options
    
    Args:
        verbose (int): Level of verbosity
        machine (str, optional): Defaults to "default". Machine name where the algorithm is run. Used to create logging filename signature
        timestamp (str, optional): Defaults to "". Time where the algorithm is run. Used to create logging filename signature
        visualize (bool, optional): Defaults to False. Set connection to visdom dashboard if true
        env_render (bool, optional): Defaults to False. Save evaluation images in directory to used later
    """

        super(MonitorParams, self).__init__(
            verbose, machine, timestamp, visualize, env_render
        )

        self.env_name = "LunarLander-v2"
        self.n_episodes = 2000
        self.max_steps_in_episode = 1000

        self.report_freq_by_episodes = 100
        self.eval_freq_by_episodes = 100
        self.eval_steps = 1000

        self.seed = 0

        self.reward_solved_criteria = 200

        self.agent_params = AgentParams(verbose, machine=machine, timestamp=timestamp)

        if self.env_render:
            self.img_dir = self.root_dir + "/imgs/"
            self.imsave = imageio.imwrite
