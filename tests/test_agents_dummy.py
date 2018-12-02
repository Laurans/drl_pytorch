import pytest
import unittest
from core.utils.params import AgentParams
from core.agents import DummyAgent, Agent
import numpy as np


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent_params = AgentParams({"verbose": 0})
        self.agent = Agent("TestAgent", self.agent_params)

    def test_act_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.agent.act(None)

    def test_learn_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.agent.learn(None)

    def test_step_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.agent.step(None, None, None, None, None)

    def test_save_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.agent.save(None)

    def test_load_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.agent.load(None)


class TestDummyAgent(unittest.TestCase):
    def setUp(self):
        self.agent_params = AgentParams({"verbose": 0})
        self.agent_params.seed = 123
        self.agent = DummyAgent(self.agent_params, 10)

    def test_seed(self):
        self.assertEqual(self.agent_params.seed, 123)

    def test_missing_params_memory(self):
        with pytest.raises(TypeError):
            DummyAgent()

    def test_act(self):
        self.assertEqual(self.agent.act(None), 0)

    def test_act2(self):
        for _ in range(2):
            action = self.agent.act(None)
        self.assertEqual(action, 4)

    def test_learn(self):
        self.assertEqual(self.agent.learn(None), None)

    def test_step(self):
        self.assertEqual(self.agent.step(None, None, None, None, None), None)

    def test_save(self):
        assert self.agent.save(None) == None

    def test_load(self):
        assert self.agent.load(None) == None
