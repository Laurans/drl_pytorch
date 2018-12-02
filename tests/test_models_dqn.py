from core.models import QNetwork_MLP
from core.utils.params import ModelParams
import numpy as np
import pytest
import unittest
import torch


class TestMLPModel(unittest.TestCase):
    def setUp(self):
        par = ModelParams({"verbose": 0})
        par.state_shape = (1,)
        par.action_dim = 1
        par.hidden_dim = [2]
        par.seed = 23
        par.hist_len = 1

        self.net = QNetwork_MLP(model_params=par)

    def test_model_seed(self):
        assert self.net.seed == 23

    def test_model_initialization_reproductible(self):
        weight = list(self.net.parameters())[0].cpu().detach().numpy()
        weight.astype(np.float32)
        print(weight)
        truth = np.array([[-0.26339746], [0.3011571]], dtype=np.float32)
        self.assertTrue((weight == truth).all())

    def test_model_forward(self):
        state = torch.from_numpy(np.array([[2.5]])).float()
        output = self.net.forward(state).cpu().detach().numpy()
        self.assertAlmostEqual(-0.54986596, output[0][0])
