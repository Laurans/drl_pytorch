from core.envs.env import Env
import gym


class GymEnv(Env):
    def __init__(self, env_params):
        super(GymEnv, self).__init__("Gym", env_params)

        self.env = gym.make(self.game)
        self.env.seed(self.seed)

    def get_state_shape(self):
        return self.env.observation_space.shape

    def get_action_size(self):
        return self.env.action_space.n

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def render(self):
        return self.env.render(mode="rgb_array")
