import pytest
import unittest
from core.utils.params import MemoryParams
from core.memories import ReplayBuffer, Memory
import numpy as np
from unittest import mock


class TestReplayBufferMethods(unittest.TestCase):
    def setUp(self):
        self.memory_params = MemoryParams({"verbose": 0})
        self.memory_params.window_length = 2
        self.memory_params.memory_size = 1000
        self.memory_params.seed = 123
        self.memory_params.combined_with_last = False

        states = np.arange(50).reshape(-1, 2)
        next_states = np.arange(2, 52).reshape(-1, 2)
        actions = np.arange(0, 25, 1) ** 2
        rewards = np.linspace(-1, 2, 25)
        terms = np.zeros(25)

        self.memory = ReplayBuffer(self.memory_params)

        for s, a, r, s2, d in zip(states, actions, rewards, next_states, terms):
            self.memory.append(s, a, r, s2, d)
            self.memory.append_recent(s, d)

    def test_length_memory(self):
        self.assertEqual(25, len(self.memory))

    def test_memory(self):
        self.assertTrue((np.array([6, 7]) == self.memory.memory[3].state).all())

    def test_get_historic_state(self):
        pass

    def test_same_sample(self):
        sample = self.memory.sample(4)[1].cpu().detach().numpy().flatten()
        self.assertEqual([1, 64, 4, 576], list(sample))

    def test_sample_size(self):
        self.assertEqual(4, len(self.memory.sample(4)[0]))

    def test_sample_return_sarsd(self):
        self.assertEqual(
            5,
            len(self.memory.sample(1)),
            msg="Must return (state, action, reward, next_state, done) tuple",
        )

    def test_missing_params_memory(self):
        with pytest.raises(TypeError):
            ReplayBuffer()

    def test_get_recent_states(self):
        assert (
            self.memory.get_recent_states(np.array([77, 88]))
            == np.array([[46, 47], [48, 49], [77, 88]])
        ).all()

    def test_zeroed_observation(self):
        s = np.array([500, 501])
        d = True
        self.memory.append_recent(s, d)

        assert (
            self.memory.get_recent_states(np.array([77, 88]))
            == np.array([[0, 0], [0, 0], [77, 88]])
        ).all()

    def test_zeroed_observation_2(self):
        s = np.array([500, 501])
        self.memory.append_recent(s, True)
        self.memory.append_recent(s, False)

        assert (
            self.memory.get_recent_states(np.array([77, 88]))
            == np.array([[0, 0], [500, 501], [77, 88]])
        ).all()

    @mock.patch("core.memories.Memory.__init__")
    def test_super_init(self, mock_super):
        ReplayBuffer(self.memory_params)
        self.assertTrue(mock_super.called)


class TestMemoryOverwrite(unittest.TestCase):
    def setUp(self):
        self.memory_params = MemoryParams({"verbose": 0})
        self.memory_params.window_length = 1
        self.memory_params.memory_size = 24
        self.memory_params.seed = 123
        self.memory_params.combined_with_last = False

        states = np.arange(50).reshape(-1, 2)
        next_states = np.arange(2, 52).reshape(-1, 2)
        actions = np.arange(0, 25, 1) ** 2
        rewards = np.linspace(-1, 2, 25)
        terms = np.zeros(25)

        self.memory = ReplayBuffer(self.memory_params)

        for s, a, r, s2, d in zip(states, actions, rewards, next_states, terms):
            self.memory.append(s, a, r, s2, d)
            self.memory.append_recent(s, d)

    def test_rolling(self):
        self.assertTrue(np.all(np.array([2, 3]) == self.memory.memory[0].state))

    def test_overwritting(self):
        self.assertTrue(np.all(np.array([48, 49]) == self.memory.memory[-1].state))


class TestMemoryCombinedWithLast(unittest.TestCase):
    def setUp(self):
        self.memory_params = MemoryParams({"verbose": 0})
        self.memory_params.window_length = 1
        self.memory_params.memory_size = 24
        self.memory_params.seed = 123
        self.memory_params.combined_with_last = True

        states = np.arange(50).reshape(-1, 2)
        next_states = np.arange(2, 52).reshape(-1, 2)
        actions = np.arange(0, 25, 1) ** 2
        rewards = np.linspace(-1, 2, 25)
        terms = np.zeros(25)

        self.memory = ReplayBuffer(self.memory_params)

        for s, a, r, s2, d in zip(states, actions, rewards, next_states, terms):
            self.memory.append(s, a, r, s2, d)
            self.memory.append_recent(s, d)

    def test_combined_with_last(self):
        s = np.array([500, 501])
        s2 = np.array([600, 601])
        a = 2
        r = 2
        d = False
        self.memory.append(s, a, r, s2, d)
        sample = self.memory.sample(10)
        assert (sample[0][-1].cpu().detach().numpy() == s).all()

    def test_len_recents_observation(self):
        assert len(self.memory.recent_observations) == 1

    def test_only_zeroed_observation(self):
        s = np.array([500, 501])
        d = True
        self.memory.append_recent(s, d)

        assert (
            self.memory.get_recent_states(np.array([77, 88]))
            == np.array([[0, 0], [77, 88]])
        ).all()


class TestReplayBufferAtStart(unittest.TestCase):
    def setUp(self):
        self.memory_params = MemoryParams({"verbose": 0})
        self.memory_params.window_length = 5
        self.memory_params.memory_size = 1000
        self.memory_params.seed = 123
        self.memory_params.combined_with_last = False
        self.memory = ReplayBuffer(self.memory_params)

    def test_only_zeroed_observation(self):
        assert (
            self.memory.get_recent_states(np.array([77, 88]))
            == np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [77, 88]])
        ).all()
