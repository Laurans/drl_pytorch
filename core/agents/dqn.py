from core.agents.agent import Agent
import random
import torch
import torch.optim as optim
import numpy as np


class MLPAgent(Agent):
    def __init__(self, agent_params, state_size, action_size, model_prototype):
        super(MLPAgent, self).__init__("MLP Agent", agent_params)

        self.action_dim = action_size
        self.model_params.state_shape = state_size
        self.model_params.action_dim = self.action_dim
        

        # Q-Network
        self.model = model_prototype(self.model_params).to(self.device)
        self.target_model = model_prototype(self.model_params).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self._update_target_model()

        self.counter_steps = 0

    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state):
        observation = state

        if self.training and self.counter_steps < self.learn_start:
            action = random.randrange(self.action_dim)
        else:
            action = self._epsilon_greedy(observation)

        return action

    def learn(self, experiences):
        pass

    def _update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _epsilon_greedy(self, observation):
        if self.training:
            self.eps = (
                self.eps_end
                + max(0, (self.eps_start - self.eps_end))
                * (self.eps_decay - max(0, self.step - self.learn_start))
                / self.eps_decay
            )
        else:
            self.eps = self.eps_eval

        if np.random.uniform() < self.eps:
            action = random.randrange(self.action_dim)

        else:
            action = self._get_action(observation)

        return action

    def _get_action(self, observation):
        observation = torch.from_numpy(np.array(state)).unsqueeze(0).to(self.device)
        q_values = self.model(Variable(observation, volatile=True)).data

        if self.use_cuda:
            action = np.argmax(q_values.cpu().numpy())
        else:
            action = np.argmax(q_values.numpy())

        return action
