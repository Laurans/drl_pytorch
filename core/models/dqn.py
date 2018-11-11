from core.models.model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(Model):
    def __init__(self, model_params):
        super(QNetwork, self).__init__("QNetwork MLP", model_params)

        self.fc1 = nn.Linear(
            np.prod(self.input_dims_0 * self.input_dims_1), self.hidden_dim
        )
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl3 = nn.ReLU()
        self.fc4 = nn.Linear(self.hidden_dim, self.output_dims)

        self.print_model()
        self.reset()

    def _init_weights(self):
        self.fc1.weight.data = nn.init.kaiming_normal_(
            self.fc1.weight.data, mode="fan_out", nonlinearity="relu"
        )
        self.fc2.weight.data = nn.init.kaiming_normal_(
            self.fc2.weight.data, mode="fan_out", nonlinearity="relu"
        )
        self.fc3.weight.data = nn.init.kaiming_normal_(
            self.fc3.weight.data, mode="fan_out", nonlinearity="relu"
        )
        self.fc4.weight.data = nn.init.kaiming_normal_(
            self.fc4.weight.data, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims_0 * self.input_dims_1)
        x = self.rl1(self.fc1(x))
        x = self.rl2(self.fc2(x))
        x = self.rl3(self.fc3(x))

        return self.fc4(x.view(x.size(0), -1))
