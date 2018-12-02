from core.envs.env import Env
from unityagents import UnityEnvironment
import numpy as np


class UnityEnv(Env):
    def __init__(self, env_params):
        super(UnityEnv, self).__init__("Unity", env_params)

        self.env = UnityEnvironment(file_name=self.game)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        self.pixels = env_params.pixels

    def get_state_shape(self):
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = self.env_info.vector_observations[0]
        return state.shape

    def get_action_size(self):
        return self.brain.vector_action_space_size

    def reset(self):
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        return self.env_info.vector_observations[0]

    def step(self, action):
        self.env_info = self.env.step(action)[self.brain_name]
        next_state = self.env_info.vector_observations[0]
        reward = self.env_info.rewards[0]
        done = self.env_info.local_done[0]
        return next_state, reward, done

    def render(self):
        if self.pixels:
            return np.squeeze(self.env_info.visual_observations[0])
        else:
            return None
