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
    par = MonitorParams(**{"verbose": 0, "machine": 'test', 'visualize': False})
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
    assert isinstance(monitor.agent, AGENT_DICT['dqn'])
