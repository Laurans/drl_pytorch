import os
import visdom

from .logger import loggerConfig


class Params:
    def __init__(self, verbose):
        self.verbose = verbose  # 0 (no set) | 1 (info) | 2 (debug)

        # signature
        self.machine = "ayrs"
        self.timestamp = "181109"

        #
        self.seed = 123
        self.visualize = True

        # prefix for saving
        self.refs = self.machine + "_" + self.timestamp
        self.root_dir = os.getcwd()

        # logging config
        self.log_name = self.root_dir + "/logs/" + self.refs + ".log"
        self.logger = loggerConfig(self.log_name, self.verbose)
        self.logger.info(":===================================:")

        if self.visualize:
            self.vis = visdom.Visdom()
            self.logger.warning("bash$: python3 -m visdom.server")
            self.logger.warning("http://localhost:8097/env/{}".format(self.refs))
