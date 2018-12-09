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

        self.hidden_layers = nn.ModuleList()

        self.input_layer = nn.Linear(
            self.input_dims_0 * np.prod(self.input_dims_1), self.hidden_dim[0]
        )

        for input_dim, output_dim in zip(self.hidden_dim[0:-1], self.hidden_dim[1:]):
            layer = nn.Linear(input_dim, output_dim)
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(self.hidden_dim[-1], self.output_dims)

        self.print_model()
        self.reset()

    def _init_weights(self) -> None:
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data = nn.init.kaiming_normal_(
                    layer.weight.data, nonlinearity="relu"
                )
        self.input_layer.weight.data = nn.init.kaiming_normal_(
            self.input_layer.weight.data, nonlinearity="relu"
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.input_layer(x))

        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))

        return self.output_layer(x)
