import dataclasses
from typing import Tuple

import numpy as np
import torch

from actor_critic.model import MLPActorCritic, MLPActorCriticModelConfig


@dataclasses.dataclass(frozen=True)
class ActorCriticConfig:
  nn_width: int
  nn_depth: int
  optimize_rate: int
  learning_rate: int
  gamma: float
  device: str
  observation_shape: Tuple
  action_size: int


class ActorCritic:
  def __init__(self, config: ActorCriticConfig):
    assert len(config.observation_shape) == 1  # Only support MLP for now.

    model_config = MLPActorCriticModelConfig(
        input_size=config.observation_shape[0], output_size=config.action_size,
        depth=config.nn_depth, width=config.nn_width)

    self._gamma = config.gamma
    self._optimize_rate = config.optimize_rate
    self._policy = MLPActorCritic(model_config)
    self._optimizer = torch.optim.Adam(self._policy.parameters(),
                                       lr=config.learning_rate)
    self._device = torch.device(config.device)

  def act(self, observation: np.ndarray) -> int:
    observation = observation.astype(np.float32)
    observation = torch.from_numpy(observation).unsqueeze(0)
    with torch.no_grad():
      action_probs, _ = self._policy(observation)

    action_distribution = torch.distributions.Categorical(action_probs)
    action = action_distribution.sample().item()

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

    if done:
      pass

  def _optimize(self):
    pass
