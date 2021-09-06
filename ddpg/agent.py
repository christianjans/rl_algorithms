import dataclasses
from os import stat
from typing import Tuple, Union
from utils.transitions import DoneTransition

import numpy as np
import torch

from utils.replay_buffer import ReplayBuffer


@dataclasses.dataclass(frozen=True)
class DDPGConfig:
  batch_size: int
  buffer_size: int
  nn_width: int
  nn_depth: int
  optimize_rate: int
  target_update_rate: int
  policy_learning_rate: float
  action_value_learning_rate: float
  polyak: float
  gamma: float
  device: str
  observation_shape: Tuple
  action_size: int


class DDPG:
  def __init__(self, config: DDPGConfig):
    assert len(config.observation_shape) == 1  # Only support MLP for now.

    self._step = 0
    self._optimize_rate = config.optimize_rate
    self._target_update_rate = config.target_update_rate
    self._gamma = config.gamma
    self._polyak = config.polyak
    self._action_size = config.action_size
    self._device = config.device
    self._batch_size = config.batch_size
    self._replay_buffer = ReplayBuffer(config.buffer_size)
    self._policy_network: torch.nn.Module = None  # TODO
    self._policy_target_network: torch.nn.Module = None  # TODO
    self._action_value_network: torch.nn.Module = None  # TODO
    self._action_value_target_network: torch.nn.Module = None  # TODO
    self._policy_optimizer = torch.optim.Adam(
        self._policy_network.parameters(), lr=config.policy_learning_rate)
    self._action_value_optimizer = torch.optim.Adam(
        self._action_value_network.parameters(),
        lr=config.action_value_learning_rate)

    # self._noise = torch.distributions.Normal()
    self._action_noise = torch.distributions.Normal(
        torch.zeros(config.action_size), torch.ones(config.action_size))

    self._update_target_networks()

  def act(self, observation: np.ndarray, explore=False) -> Tuple:
    observation = observation.astype(np.float32)
    observation = torch.from_numpy(observation).unsqueeze(dim=0)
    with torch.no_grad():
      action_tensor = self._policy_network(observation)
    action_tensor += self._action_noise.sample()

    return tuple(action_tensor.tolist())

  def step(self,
           observation: np.ndarray,
           action: Tuple,
           reward: float,
           next_observation: np.ndarray,
           done: bool):
    self._step += 1

    observation = torch.from_numpy(observation.astype(np.float32))
    action = torch.tensor(action)
    reward = torch.tensor(reward)
    next_observation = torch.tensor(next_observation.astype(np.float32))
    done = torch.tensor(1) if done else torch.tensor(0)

    self._replay_buffer.add(DoneTransition(observation, action, reward, next_observation, done))

    if self._step % self._optimize_rate == 0:
      # Optimize the models.
      self._optimize()

    if self._step % self._target_update_rate == 0:
      # Update the target q network.
      self._update_target_networks()

  def _optimize(self):
    if len(self._replay_buffer) < self._batch_size:
      return

    # Sample data from the replay buffer.
    transitions = self._replay_buffer.sample(self._batch_size)
    batch = DoneTransition(*zip(*transitions))

    # Prepare the data.
    state_batch = torch.stack(batch.observation).to(self._device)
    action_batch = torch.stack(batch.action).to(self._device)
    reward_batch = torch.stack(batch.rewards).to(self._device)
    next_state_batch = torch.stack(batch.next_observation).to(self._device)
    done_batch = torch.stack(batch.done).to(self._device)
    next_action_batch = self._policy_target_network(next_state_batch).detach()  # TODO: Is detach() needed here?

    # Optimize the action value network.
    action_values = self._action_value_network(state_batch, action_batch)
    next_action_values = self._action_value_target_network(next_state_batch, next_action_batch).detach()
    action_value_targets = reward_batch + self._gamma * (1 - done_batch) * next_action_values

    loss_function = torch.nn.MSELoss()
    loss = loss_function(action_values, action_value_targets)

    self._action_value_optimizer.zero_grad()
    loss.backward()
    self._action_value_optimizer.step()

    # Turn off gradient calculations for the action value network for policy network optimization.
    # for parameter in self._action_value_network.parameters():
    #   parameter.requires_grad = False  # TODO: Is this correct?
    self._action_value_network.requires_grad_(False)  # TODO: Is this correct?

    # Optimize the policy network.
    policy_action_batch = self._policy_network(state_batch)
    policy_gradient = self._action_value_network(state_batch, policy_action_batch)
    policy_gradient = -policy_gradient

    self._policy_optimizer.zero_grad()
    policy_gradient.backward()
    self._policy_optimizer.step()

    # Turn on gradient calculations for the action value network.
    # for parameter in self._action_value_network.parameters():
    #   parameter.requires_grad = True  # TODO: Is this correct?
    self._action_value_network.requires_grad_(True)  # TODO: Is this correct?

    # Update the action value target network parameters using Polyak averaging.
    with torch.no_grad():
      for parameter, target_parameter in zip(self._action_value_network.parameters(), self._action_value_target_network.parameters()):
        # Perform in-place operations.
        target_parameter.mul_(self._polyak)
        target_parameter.add_((1 - self._polyak) * parameter)

    # Update the policy target network parameters using Polyak averaging.
    with torch.no_grad():
      for parameter, target_parameter in zip(self._policy_network.parameters(), self._policy_target_network.parameters()):
        # Perform in-place operations.
        target_parameter.mul_(self._polyak)
        target_parameter.add((1 - self._polyak) * parameter)

  def _update_target_networks(self):
    self._policy_target_network.load_state_dict(self._policy_network.state_dict())
    self._policy_target_network.eval()
    self._action_value_target_network.load_state_dict(self._action_value_network.state_dict())
    self._action_value_target_network.eval()
