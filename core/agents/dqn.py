from core.agents.agent import Agent
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from core.memories.replaybuffer import ReplayBuffer


from core.utils.params import AgentParams
from core.models.model import Model
from core.memories.memory import Memory
from numpy import float64, ndarray
from typing import Tuple, Type, Union
from numpy import float64, int64, ndarray

import pdb


class MLPAgent(Agent):
    def __init__(
        self,
        agent_params: AgentParams,
        state_shape: Tuple[int],
        action_size: int,
        model_prototype: Type[Model],
        memory_prototype: Type[Memory],
    ) -> None:
        super(MLPAgent, self).__init__("MLP Agent", agent_params)

        self.action_dim = action_size
        self.model_params.state_shape = state_shape
        self.model_params.action_dim = self.action_dim

        # Q-Network
        self.model = model_prototype(self.model_params).to(self.device)
        self.target_model = model_prototype(self.model_params).to(self.device)
        self.optimizer = self.optim(self.model.parameters(), **self.optim_params)

        self._update_target_model()

        # Memory
        self.memory = memory_prototype(self.memory_params)

        self.t_step = 0
        random.seed(self.seed)
        np.random.seed(self.seed)

    def step(
        self,
        state: ndarray,
        action: int,
        reward: Union[float64, int],
        next_state: ndarray,
        done: bool,
    ) -> None:

        s = self.memory.get_recent_states(state).flatten()
        s2 = self.memory.get_recent_states(state, next_state).flatten()
        self.memory.append(s, action, float(reward), s2, done)
        self.memory.append_recent(state, done)
        self.t_step = (self.t_step + 1) % self.learn_every

    def act(self, observation: ndarray) -> int:
        observation = self.memory.get_recent_states(observation).flatten()

        if self.training:
            action = self._epsilon_greedy(observation)
        else:
            action, _ = self.get_raw_actions(observation)

        return action

    def learn(self) -> None:
        if len(self.memory) >= self.batch_size:
            self.model.train()
            experiences = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = experiences

            Q_targets_next = (
                self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
            )
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

            Q_expected = self.model(states).gather(1, actions)

            loss = F.mse_loss(Q_expected, Q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-self.clip_grad, self.clip_grad)

            self.optimizer.step()
            self._soft_update_target_model()

            return loss.cpu().detach().numpy()

    def save(self, checkpoint=""):
        if checkpoint == "":
            checkpoint = f"{self.model_dir}{self.agent_name}.pth"

        torch.save(self.model.state_dict(), checkpoint)

    def load(self, checkpoint=""):
        if checkpoint == "":
            checkpoint = f"{self.model_dir}{self.agent_name}.pth"

        self.model.load_state_dict(torch.load(checkpoint))

    def _update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def _soft_update_target_model(self) -> None:
        for target_param, local_param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def update_epsilon(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def _epsilon_greedy(self, observation: ndarray) -> Union[int64, int]:
        if np.random.uniform() < self.eps:
            action = random.randrange(self.action_dim)

        else:
            action, _ = self.get_raw_actions(observation)

        return action

    def get_raw_actions(self, observation: ndarray) -> int64:
        observation = (
            torch.from_numpy(np.array(observation)).unsqueeze(0).float().to(self.device)
        )

        with torch.no_grad():
            self.model.eval()
            q_values = self.model(observation).data

        if self.use_cuda:
            q_values = q_values.cpu().numpy()
        else:
            q_values = q_values.numpy()

        return np.argmax(q_values), q_values
