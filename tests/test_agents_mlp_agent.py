import pytest
import unittest
from core.utils.params import AgentParams
from core.models.dqn_mlp import QNetwork_MLP
from core.memories.replaybuffer import ReplayBuffer
from core.agents import MLPAgent
import numpy as np


@pytest.fixture
def agent():
    par = AgentParams({"verbose": 0})
    par.seed = 123
    return MLPAgent(par, (4,), 2, QNetwork_MLP, ReplayBuffer)


def test_seed_123(agent):
    assert agent.seed == 123


def test_step_return_none(agent):
    state = np.zeros((4,))
    action = 0
    reward = 0
    next_state = np.ones((4,))
    done = False
    assert agent.step(state, action, reward, next_state, done) == None


def test_step_add_in_memory(agent):
    state = np.zeros((4,))
    action = 0
    reward = 0
    next_state = np.ones((4,))
    done = False
    agent.step(state, action, reward, next_state, done)
    assert len(agent.memory) == 1
