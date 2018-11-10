import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, model_name, model_params):
        super(Model, self).__init__()

        self.logger = model_params.logger
        self.logger.info("-----------------------------[ {} ]------------------".format(model_name))

        self.hidden_dim = model_params.hidden_dim
        self.use_cuda = model_params.use_cuda
        self.seed = torch.manual_seed(model_params.seed)
        self.dtype = model_params.dtype

        self.input_dims_0 = model_params.hist_len
        self.input_dims_1 = model_params.state_shape
        self.output_dims = model_params.action_dim

    def _init_weights(self):
        raise NotImplementedError("not implemented init weights")

    def reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.logger.warning("(Re)Initialize model")

    def print_model(self):
        self.logger.info(self)