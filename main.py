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
from core.memories.replaybuffer import ReplayBuffer

agent = MLPAgent(
    agent_params=AgentParams(verbose=1),
    state_size=env.observation_space.shape,
    action_size=env.action_space.n,
    model_prototype=QNetwork,
    memory_prototype = ReplayBuffer
)

max_t=1000
scores_window = deque(maxlen=100)
scores = []
n_episodes=2

for i_episode in range(1, n_episodes+1):
    state = env.reset()
    score = 0
    for t in range(max_t):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done:
            break
    
    scores_window.append(score)
    scores.append(score)
    agent.update_epsilon()

    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

    if i_episode % 100 == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    if np.mean(scores_window)>=200.0:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        break