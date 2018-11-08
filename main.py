import gym
import random
import torch
import numpy as np
from collections import deque

env = gym.make("LunarLander-v2")
env.seed(0)
print("State shape: ", env.observation_space.shape)
print("Number of actions: ", env.action_space.n)


from core.agents.dummy import DummyAgent

agent = DummyAgent(
    state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0
)

state = env.reset()
for j in range(200):
    action = agent.act(state)
    state, reward, done, _ = env.step(action)
    print(reward)
    if done:
        break

env.close()
