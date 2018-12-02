import pytest
import numpy as np
from core.utils.params import EnvParams
from core.envs.gym import GymEnv


@pytest.fixture
def env_gym():
    par = EnvParams({"verbose": 0})
    par.seed = 123
    par.game = "LunarLander-v2"
    return GymEnv(par)


def test_env_state_shape(env_gym):
    assert env_gym.get_state_shape() == (8,)
