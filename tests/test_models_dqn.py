from core.models import QNetwork
from core.utils.params import ModelParams
import numpy as np
import pytest
import unittest
import torch


class TestMLPModel(unittest.TestCase):
    def setUp(self):
        par = ModelParams(0)
        par.state_shape = (1,)
        par.action_dim = 1
        par.hidden_dim = 2
        par.seed = 23

        self.net = QNetwork(model_params=par)

    def test_model_initialization_reproductible(self):
        weight = list(self.net.parameters())[0].cpu().detach().numpy()
        weight.astype(np.float32)
        truth = np.array([[-1.7663301], [0.8095292]], dtype=np.float32)
        self.assertTrue((weight == truth).all())

    def test_model_forward(self):
        state = torch.from_numpy(np.array([[2.5]])).float()
        output = self.net.forward(state).cpu().detach().numpy()
        self.assertAlmostEqual(2.8903606, output[0][0])
