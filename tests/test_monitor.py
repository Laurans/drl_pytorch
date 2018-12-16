import pytest
from core.utils.params import MonitorParams
from core.monitors import Monitor
from core.agents import DummyAgent
from core.envs.gym import GymEnv
import numpy as np
from core.agents import AGENT_DICT
from core.utils import MonitorParams
from core.models import MODEL_DICT
from core.memories import MEMORY_DICT
from core.envs import ENV_DICT


@pytest.fixture
def monitor():
    par = MonitorParams(**{"verbose": 0, "machine": "test", "visualize": False})
    par.seed = 123

    monitor = Monitor(
        monitor_param=par,
        agent_prototype=AGENT_DICT[par.agent_type],
        model_prototype=MODEL_DICT[par.model_type],
        memory_prototype=MEMORY_DICT[par.memory_type],
        env_prototype=ENV_DICT[par.env_type],
    )

    return monitor


def test_init(monitor):
    assert isinstance(monitor.agent, AGENT_DICT["dqn"])


def test_train_on_episode(monitor):
    ep_reward, ep_steps, losses = monitor._train_on_episode()
    assert ep_reward == pytest.approx(-154.71, 0.1)


def test_train(monitor):
    monitor.train_n_episodes = 1
    monitor.visualize = True
    monitor.report_freq = 1

    with pytest.raises(AttributeError):
        monitor.train()


def test_train_resolve(monitor):
    monitor.train_n_episodes = 1
    monitor.reward_solved_criteria = -10000
    monitor.output_filename = "test.pth"
    monitor.train()
    assert monitor.summaries["eval_steps_avg"]["log"][0][0] == 103


def test_train_evaluation(monitor):
    monitor.train_n_episodes = 1
    monitor.eval_during_training = True
    monitor.eval_freq = 1
    monitor.train()
    assert monitor.summaries["eval_state_values"]["log"][-1][1] == pytest.approx(
        -0.122016974, 0.001
    )
