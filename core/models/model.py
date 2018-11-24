import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from core.utils.params import ModelParams


class Model(nn.Module):
    def __init__(self, model_name: str, model_params: ModelParams) -> None:
        super(Model, self).__init__()

        self.logger = model_params.logger

        self.hidden_dim = model_params.hidden_dim
        self.use_cuda = model_params.use_cuda
        self.seed = model_params.seed
        torch.manual_seed(self.seed)
        self.device = model_params.device

        self.input_dims_0 = model_params.hist_len
        self.input_dims_1 = model_params.state_shape
        self.output_dims = model_params.action_dim

        self.logger.info(
            f"-----------------------------[ {model_name} w/ seed {self.seed} on device {self.device} ]------------------".format(model_name)
        )

    def _init_weights(self):
        raise NotImplementedError("not implemented init weights")

    def reset(self) -> None:
        self._init_weights()
        self.logger.warning("(Re)Initialize model")

    def print_model(self) -> None:
        self.logger.info(self)
