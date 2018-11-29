from core.monitors import Monitor
from core.agents import AGENT_DICT
from core.utils import MonitorParams
from core.models import MODEL_DICT
from core.memories import MEMORY_DICT
from core.envs import ENV_DICT

options = MonitorParams(verbose=1, visualize=True, env_render=True)#, machine='unity', timestamp='0643')

monitor = Monitor(
    monitor_param=options,
    agent_prototype=AGENT_DICT[options.agent_type],
    model_prototype=MODEL_DICT[options.model_type],
    memory_prototype=MEMORY_DICT[options.memory_type],
    env_prototype=ENV_DICT[options.env_type]
)

monitor.train()
