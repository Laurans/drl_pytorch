from core.models.model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from core.utils.params import ModelParams
from torch import Tensor


class QNetwork_MLP(Model):
    def __init__(self, model_params: ModelParams) -> None:
        super(QNetwork_MLP, self).__init__("QNetwork MLP", model_params)

        self.fc1 = nn.Linear(
            np.prod(self.input_dims_0 * self.input_dims_1), self.hidden_dim
        )
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dims)

        self.print_model()
        self.reset()

    def _init_weights(self) -> None:
        self.fc1.weight.data = nn.init.kaiming_normal_(
            self.fc1.weight.data, mode="fan_out", nonlinearity="relu"
        )
        self.fc2.weight.data = nn.init.kaiming_normal_(
            self.fc2.weight.data, mode="fan_out", nonlinearity="relu"
        )
        self.fc3.weight.data = nn.init.kaiming_normal_(
            self.fc3.weight.data, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
