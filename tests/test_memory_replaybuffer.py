import pytest
import unittest
from core.utils.params import MemoryParams
from core.memories import ReplayBuffer, Memory
import numpy as np
from unittest import mock


class TestReplayBufferMethods(unittest.TestCase):
    def setUp(self):
        self.memory_params = MemoryParams(0)
        self.memory_params.window_length = 2
        self.memory_params.memory_size = 1000

        states = np.arange(50).reshape(-1, 2)
        next_states = np.arange(2, 52).reshape(-1, 2)
        actions = np.arange(0, 25, 1) ** 2
        rewards = np.linspace(-1, 2, 25)
        terms = np.zeros(25)

        self.memory = ReplayBuffer(self.memory_params)

        for s, a, r, s2, d in zip(states, actions, rewards, next_states, terms):
            self.memory.append(s, a, r, s2, d)

    def test_length_memory(self):
        self.assertEqual(25, len(self.memory))

    def test_memory(self):
        self.assertTrue((np.array([6, 7]) == self.memory.memory[3].state).all())

    def test_get_historic_state(self):
        pass

    def test_same_sample(self):
        sample = self.memory.sample(4)[1].cpu().detach().numpy().flatten()
        self.assertEqual([1,64,4,576], list(sample))

    def test_sample_size(self):
        self.assertEqual(4, len(self.memory.sample(4)[0]))

    def test_sample_return_sarsd(self):
        self.assertEqual(5, len(self.memory.sample(1)), msg="Must return (state, action, reward, next_state, done) tuple")


    def test_missing_params_memory(self):
        with pytest.raises(TypeError):
            ReplayBuffer()

    @mock.patch("core.memories.Memory.__init__")
    def test_super_init(self, mock_super):
        ReplayBuffer(self.memory_params)
        self.assertTrue(mock_super.called)


class TestMemoryOverwrite(unittest.TestCase):
    def setUp(self):
        self.memory_params = MemoryParams(0)
        self.memory_params.window_length = 1
        self.memory_params.memory_size = 24

        states = np.arange(50).reshape(-1, 2)
        next_states = np.arange(2, 52).reshape(-1, 2)
        actions = np.arange(0, 25, 1) ** 2
        rewards = np.linspace(-1, 2, 25)
        terms = np.zeros(25)

        self.memory = ReplayBuffer(self.memory_params)

        for s, a, r, s2, d in zip(states, actions, rewards, next_states, terms):
            self.memory.append(s, a, r, s2, d)

    def test_rolling(self):
        self.assertTrue(np.all(np.array([2, 3]) == self.memory.memory[0].state))

    def test_overwritting(self):
        self.assertTrue(np.all(np.array([48, 49]) == self.memory.memory[-1].state))
