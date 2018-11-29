from core.agents.agent import Agent
from core.agents.dqn import MLPAgent
from core.agents.dummy import DummyAgent

AGENT_DICT = {"dummy": DummyAgent, "dqn": MLPAgent}
