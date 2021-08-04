# Implementation inspired by
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.

import dataclasses
from typing import Tuple

import numpy as np
import torch

from dqn.model import MLPModel, MLPModelConfig
from utils.replay_buffer import ReplayBuffer
from utils.transitions import DoneTransition


@dataclasses.dataclass(frozen=True)
class DQNConfig:
  batch_size: int
  buffer_size: int
  nn_width: int
  nn_depth: int
  optimize_rate: int
  target_update_rate: int
  learning_rate: float
  gamma: float
  epsilon_start: float
  epsilon_end: float
  epsilon_decay: int
  device: str
  observation_shape: Tuple
  action_size: int


class DQN:
  def __init__(self, config: DQNConfig):
    assert len(config.observation_shape) == 1  # Only support MLP for now.

    model_config = MLPModelConfig(input_size=config.observation_shape[0],
                                  output_size=config.action_size,
                                  depth=config.nn_depth,
                                  width=config.nn_width)

    self._step = 0
    self._action_size = config.action_size
    self._epsilon_start = config.epsilon_start
    self._epsilon_end = config.epsilon_end
    self._epsilon_decay = config.epsilon_decay
    self._gamma = config.gamma
    self._batch_size = config.batch_size
    self._optimize_rate = config.optimize_rate
    self._target_update_rate = config.target_update_rate
    self._replay_buffer = ReplayBuffer(config.buffer_size)
    self._device = torch.device(config.device)
    self._policy_network = MLPModel(model_config)
    self._target_network = MLPModel(model_config)
    self._optimizer = torch.optim.Adam(self._policy_network.parameters(),
                                       lr=config.learning_rate)

    self._update_target_network()

  def act(self, observation: np.ndarray, explore=False) -> int:
    epsilon = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * np.exp(-1.0 * self._step / self._epsilon_decay)

    if explore and np.random.uniform(0, 1) < epsilon:
      action = np.random.randint(self._action_size)
      return action
    else:
      observation = observation.astype(np.float32)
      observation = torch.from_numpy(observation).unsqueeze(dim=0)
      with torch.no_grad():
        network_output = self._policy_network(observation)
        action = torch.argmax(network_output, dim=1).item()
        return action

  def step(self,
           observation: np.ndarray,
           action,
           reward: float,
           next_observation: np.ndarray,
           done: bool):
    self._step += 1

    observation = torch.from_numpy(observation.astype(np.float32))
    action = torch.tensor([action])
    reward = torch.tensor([reward])
    next_observation = torch.tensor(next_observation.astype(np.float32))
    done = torch.tensor([done])

    self._replay_buffer.add(DoneTransition(observation, action, reward, next_observation, done))

    if self._step % self._optimize_rate == 0:
      self._optimize()

    if self._step % self._target_update_rate == 0:
      self._update_target_network()

  def _optimize(self):
    if len(self._replay_buffer) < self._batch_size:
      return

    transitions = self._replay_buffer.sample(self._batch_size)
    batch = DoneTransition(*zip(*transitions))

    state_batch = torch.stack(batch.observation).to(self._device)
    action_batch = torch.stack(batch.action).to(self._device)
    reward_batch = torch.stack(batch.reward).to(self._device)
    next_state_batch = torch.stack(batch.next_observation).to(self._device)
    not_done_batch = ~torch.stack(batch.done).to(self._device)

    state_action_values = self._policy_network(state_batch).gather(dim=1, index=action_batch)
    next_state_values = self._target_network(next_state_batch).max(dim=1, keepdim=True)[0].detach()
    next_state_values = torch.where(not_done_batch, next_state_values, torch.zeros_like(next_state_values))
    expected_state_action_values = reward_batch + self._gamma * next_state_values

    loss_function = torch.nn.MSELoss()
    loss = loss_function(state_action_values, expected_state_action_values)

    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()

  def _update_target_network(self):
    self._target_network.load_state_dict(self._policy_network.state_dict())
    self._target_network.eval()


if __name__ == "__main__":
  import gym

  EPISODES = 200

  env = gym.make("CartPole-v0")

  assert isinstance(env.action_space, gym.spaces.Discrete)

  dqn_config = DQNConfig(batch_size=128,
                         buffer_size=10000,
                         nn_width=16,
                         nn_depth=4,
                         optimize_rate=1,
                         target_update_rate=10,
                         learning_rate=0.001,
                         gamma=0.999,
                         epsilon_start=0.9,
                         epsilon_end=0.05,
                         epsilon_decay=200,
                         device="cpu",
                         observation_shape=env.observation_space.shape,
                         action_size=env.action_space.n)
  dqn_agent = DQN(dqn_config)

  for episode_index in range(EPISODES):
    episode_return = 0.0
    done = False
    observation = env.reset()

    while not done:
      action = dqn_agent.act(observation, explore=True)
      next_observation, reward, done, _ = env.step(action)
      dqn_agent.step(observation, action, reward, next_observation, done)
      episode_return += reward
      observation = next_observation

    print(f"Return in episode #{episode_index + 1}: {episode_return}")
