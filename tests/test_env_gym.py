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


def test_env_reset(env_gym):
    state = env_gym.reset()
    assert state.sum() == pytest.approx(1.3239913, 0.001)


def test_env_step(env_gym):
    env_gym.reset()
    s2, r, d = env_gym.step(0)
    assert s2.sum() == pytest.approx(1.292151, 0.001)


def test_env_render(env_gym):
    img = env_gym.render()
    assert img.ndim == 3
