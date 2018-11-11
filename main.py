import gym
import random
import torch
import numpy as np
from collections import deque


env = gym.make("LunarLander-v2")
env.seed(0)


from core.agents.dqn import MLPAgent
from core.utils.params import AgentParams
from core.models.dqn import QNetwork

agent = MLPAgent(
    agent_params=AgentParams(verbose=1),
    state_size=env.observation_space.shape,
    action_size=env.action_space.n,
    model_prototype=QNetwork,
)

state = env.reset()
for j in range(200):
    action = agent.act(state)
    state, reward, done, _ = env.step(action)
    print(reward)
    if done:
        break

env.close()
