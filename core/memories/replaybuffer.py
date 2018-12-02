from core.memories.memory import Memory
import random
from collections import namedtuple, deque
import numpy as np
import torch

from core.utils.params import MemoryParams
from numpy import float64, ndarray
from typing import Union


class ReplayBuffer(Memory):
    def __init__(self, memory_params: MemoryParams) -> None:
        self.combined_with_last = memory_params.combined_with_last
        prefix = "Combined " if self.combined_with_last else ""
        super(ReplayBuffer, self).__init__(f"{prefix}Replay Buffer", memory_params)

    def append(
        self,
        observation: ndarray,
        action: int,
        reward: float,
        next_observation: ndarray,
        terminal: bool,
    ) -> None:

        self.memory.append(
            self.experience(observation, action, reward, next_observation, terminal)
        )

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for e in experiences:
            if e is not None:
                states.append(e.state)
                actions.append(e.action)
                rewards.append(e.reward)
                next_states.append(e.next_state)
                dones.append(e.done)

        if self.combined_with_last:
            last_experience = self.memory[-1]

            states.append(last_experience.state)
            actions.append(last_experience.action)
            rewards.append(last_experience.reward)
            next_states.append(last_experience.next_state)
            dones.append(last_experience.done)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = (
            torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        )

        return (states, actions, rewards, next_states, dones)
