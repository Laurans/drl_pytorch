from collections import namedtuple, deque

from core.utils.params import MemoryParams
from numpy import float64, ndarray
from typing import Union
import random

import numpy as np


class Memory:
    def __init__(self, memory_name: str, memory_params: MemoryParams) -> None:

        self.logger = memory_params.logger

        self.window_length = memory_params.window_length

        self.recent_observations = deque(maxlen=self.window_length)
        self.recent_terminals = deque(maxlen=self.window_length)
        self.ignore_episode_end = False
        self.memory_size = memory_params.memory_size
        self.experience = memory_params.experience

        self.device = memory_params.device

        self.memory = deque(maxlen=self.memory_size)

        self.seed = memory_params.seed
        random.seed(self.seed)
        self.logger.info(
            f"-----------------------------[ {memory_name} w/ seed {self.seed} ]------------------"
        )

    def sample(self, batch_size):
        raise NotImplementedError("not implemented sample method in memory")

    def append_recent(self, observation: ndarray, terminal: bool) -> None:
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_states(self, current_observation, next_observation=None):
        if current_observation.ndim == 1:
            current_observation = current_observation.reshape(1, -1)
        past_observations = list()
        if next_observation is not None:
            if next_observation.ndim == 1:
                next_observation = next_observation.reshape(1, -1)
            past_observations.append(next_observation)
        past_observations.append(current_observation)

        flag = False
        for state, terminal in zip(
            list(self.recent_observations)[::-1], list(self.recent_terminals)[::-1]
        ):
            if terminal or flag:
                past_observations.append(np.zeros_like(current_observation))
                flag = True
            else:
                past_observations.append(state)

        while len(past_observations) < self.window_length + 1:
            past_observations.append(np.zeros_like(current_observation))

        past_observations = past_observations[: self.window_length + 1]

        return np.vstack(past_observations[::-1])

    def __len__(self):
        return len(self.memory)
