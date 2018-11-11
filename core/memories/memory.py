from collections import namedtuple, deque

from core.utils.params import MemoryParams
from numpy import float64, ndarray
from typing import Union


class Memory:
    def __init__(self, memory_name: str, memory_params: MemoryParams) -> None:

        self.logger = memory_params.logger
        self.logger.info(
            "-----------------------------[ {} ]------------------".format(memory_name)
        )

        self.window_length = memory_params.window_length

        self.recent_observations = deque(maxlen=self.window_length)
        self.recent_terminals = deque(maxlen=self.window_length)
        self.ignore_episode_end = False
        self.memory_size = memory_params.memory_size
        self.experience = memory_params.experience

        self.device = memory_params.device

        self.memory = deque(maxlen=self.memory_size)

    def sample(self, batch_size):
        raise NotImplementedError("not implemented sample method in memory")

    def append(
        self,
        observation: ndarray,
        action: int,
        reward: float,
        next_observation: ndarray,
        terminal: bool,
    ) -> None:
        self.recent_observations.append(observation)
        self.recent_observations.append(terminal)

    def get_recent_state(self, current_observation):
        past_observations = list(past_observations)
        past_observation.append(current_observation)
        # TODO zeroed observation

    def __len__(self):
        return len(self.memory)
