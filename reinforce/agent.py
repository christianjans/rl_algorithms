# Implementation inspired by
# https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63

import dataclasses

from reinforce.model import MLPDistributionModel, MLPDistributionModelConfig
from typing import List, Tuple

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class ReinforceConfig:
  nn_width: int
  nn_depth: int
  optimize_rate: int
  learning_rate: float
  gamma: float
  device: str
  observation_shape: Tuple
  action_size: int


class Reinforce:
  def __init__(self, config: ReinforceConfig):
    assert len(config.observation_shape) == 1  # Only support MLP for now.

    model_config = MLPDistributionModelConfig(
        input_size=config.observation_shape[0], output_size=config.action_size,
        depth=config.nn_depth, width=config.nn_width)

    self._gamma = config.gamma
    self._optimize_rate = config.optimize_rate
    self._policy = MLPDistributionModel(model_config)
    self._optimizer = torch.optim.Adam(self._policy.parameters(),
                                       lr=config.learning_rate)
    self._device = torch.device(config.device)
    self._observation_sequences: List[List[np.ndarray]] = []
    self._action_sequences: List[List[int]] = []
    self._reward_sequences: List[List[float]] = []
    self._episode = 1

    self._reset_experience()

  def act(self, observation: np.ndarray, explore: bool = False):
    # print(f"observation: {observation}")
    observation = observation.astype(np.float32)
    observation = torch.from_numpy(observation).unsqueeze(0)
    with torch.no_grad():
      probs = self._policy(observation)
    # print(f"probs: {probs}")

    distribution = torch.distributions.Categorical(probs)
    action = distribution.sample().item()

    return action

  def step(self,
           observation: np.ndarray,
           action: int,
           reward: float,
           next_observation: np.ndarray,
           done: bool):
    observation = torch.from_numpy(observation.astype(np.float32))
    action = torch.tensor(action)
    reward = torch.tensor(reward)

    self._observation_sequences[-1].append(observation)
    self._action_sequences[-1].append(action)
    self._reward_sequences[-1].append(reward)

    if done:
      self._episode += 1

      # TODO: Clean this up.
      if self._episode % self._optimize_rate == 0:
        self._optimize()
        self._reset_experience()
      else:
        self._observation_sequences.append([])
        self._action_sequences.append([])
        self._reward_sequences.append([])

  def _optimize(self):
    # Calculate the return for each trajectory.
    returns = []
    for reward_sequence in self._reward_sequences:
      trajectory_return, discount = 0.0, 1.0

      for reward in reward_sequence:
        trajectory_return += discount * reward
        discount *= self._gamma

      returns.append(trajectory_return)
    # print(returns)
    returns = torch.tensor(returns)
    # print(returns)
    # print(torch.mean(returns))
    # print(torch.std(returns))
    # NOTE: Only use if the batch size > 1.
    returns = (returns - torch.mean(returns)) / (torch.std(returns) + 1e-6)

    # print(f"returns: {returns}")

    # Calculate the policy gradient.
    policy_gradient = []
    for observation_sequence, action_sequence, trajectory_return in zip(self._observation_sequences, self._action_sequences, returns):
      observation_batch = torch.stack(observation_sequence)
      action_batch = torch.stack(action_sequence)

      # print(f"observation_batch: {observation_batch}")
      # print(f"action_batch: {action_batch}")

      probs = self._policy(observation_batch)
      # print(f"probs: {probs}")
      distribution = torch.distributions.Categorical(probs)
      action_log_probs = distribution.log_prob(action_batch)

      # print(f"action_log_probs: {action_log_probs}")
      # print(f"trajectory_return: {trajectory_return}")

      policy_gradient.append(torch.sum(action_log_probs) * trajectory_return)
    # print(f"policy_gradient: {policy_gradient}")
    policy_gradient = torch.stack(policy_gradient)
    policy_gradient = torch.sum(policy_gradient)
    policy_gradient = -policy_gradient  # TODO: Is this correct?

    # print(f"policy_gradient: {policy_gradient}")

    self._optimizer.zero_grad()
    policy_gradient.backward()
    self._optimizer.step()

  def _reset_experience(self):
    self._observation_sequences.clear()
    self._action_sequences.clear()
    self._reward_sequences.clear()
    self._observation_sequences.append([])
    self._action_sequences.append([])
    self._reward_sequences.append([])
