import gym
import random
import torch
import numpy as np
from collections import deque


env = gym.make("LunarLander-v2")
env.seed(0)


from core.agents import MLPAgent
from core.utils import AgentParams
from core.models import QNetwork_MLP
from core.memories import ReplayBuffer

agent = MLPAgent(
    agent_params=AgentParams(verbose=0),
    state_size=env.observation_space.shape,
    action_size=env.action_space.n,
    model_prototype=QNetwork_MLP,
    memory_prototype=ReplayBuffer,
)

max_t = 1000
scores_window = deque(maxlen=100)
scores = []
n_episodes = 200

for i_episode in range(1, n_episodes + 1):
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

    print(
        "\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_window)),
        end="",
    )

    if i_episode % 100 == 0:
        print()
        agent.logger.info(
            "Episode {}\tAverage Score: {:.2f}".format(
                i_episode, np.mean(scores_window)
            )
        )

    if np.mean(scores_window) >= 200.0:
        print()
        agent.logger.info(
            "Environment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                i_episode - 100, np.mean(scores_window)
            )
        )
        break
