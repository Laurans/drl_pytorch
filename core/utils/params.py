import os
import visdom
import torch

from .logger import loggerConfig


class Params:
    def __init__(self, verbose, name):
        self.verbose = 1  # 0 (no set) | 1 (info) | 2 (debug)

        # signature
        self.machine = 'machine'
        self.timestamp = "181109"

        #
        self.seed = 123
        self.visualize = True

        # prefix for saving
        self.refs = self.machine + "_" + self.timestamp
        self.root_dir = os.getcwd()

        # logging config
        self.log_name = self.root_dir + "/logs/" + self.refs + ".log"
        self.logger = loggerConfig(self.log_name, self.verbose, name)
        self.logger.info(":===================================:")

        if self.visualize:
            self.vis = visdom.Visdom()
            self.logger.info("bash$: python3 -m visdom.server")
            self.logger.info("http://localhost:8097/env/{}".format(self.refs))

        self.use_cuda = torch.cuda.is_available()
        self.dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor


class ModelParams(Params):
    def __init__(self, verbose):
        super(ModelParams, self).__init__(verbose, 'model')

        self.hist_len = 1
        self.hidden_dim = 16

        
class AgentParams(Params):
    def __init__(self, verbose):
        super(AgentParams, self).__init__(verbose, 'agent')

        self.model_params = ModelParams(verbose)