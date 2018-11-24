import pytest
import unittest
from core.utils.params import AgentParams
from core.models.dqn_mlp import QNetwork_MLP
from core.memories.replaybuffer import ReplayBuffer
from core.agents import MLPAgent
import numpy as np

class TestMLPAgent(unittest.TestCase):
    def setUp(setf):
        self.AgentParams = AgentParams(0)
        self.agent = MLPAgent(AgentParams, (4,), 2, QNetwork_MLP, ReplayBuffer)

        