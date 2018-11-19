from core.monitors import Monitor
from core.agents import MLPAgent
from core.utils import MonitorParams
from core.models import QNetwork_MLP
from core.memories import ReplayBuffer

monitor = Monitor(
    monitor_param=MonitorParams(verbose=1),
    agent_prototype=MLPAgent,
    model_prototype=QNetwork_MLP,
    memory_prototype=ReplayBuffer,
)

monitor.train()
