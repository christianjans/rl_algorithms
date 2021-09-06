import dataclasses
from typing import Tuple

import torch


@dataclasses.dataclass
class MLPActorCriticModelConfig:
  input_size: int
  output_size: int
  depth: int
  width: int


class MLPActorCritic(torch.nn.Module):
  def __init__(self, config: MLPActorCriticModelConfig):
    super(MLPActorCriticModelConfig, self).__init__()

    self._layers = torch.nn.ModuleList()

    # Input layer.
    self._layers.append(torch.nn.Linear(config.input_size, config.width))
    self._layers.append(torch.nn.ReLU())

    # Torso layers.
    for _ in range(config.depth):
      self._layers.append(torch.nn.Linear(config.width, config.width))
      self._layers.append(torch.nn.ReLU())

    # Actor head.
    self._actor_linear1 = torch.nn.Linear(config.width, config.width)
    self._actor_relu = torch.nn.ReLU()
    self._actor_linear2 = torch.nn.Linear(config.width, config.output_size)
    self._actor_softmax = torch.nn.Softmax(dim=1)

    # Critic head.
    self._critic_linear1 = torch.nn.Linear(config.width, config.width)
    self._critic_relu = torch.nn.ReLU()
    self._critic_linear2 = torch.nn.Linear(config.width, 1)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    for layer in self._layers:
      x = layer(x)

    action_probs = self._actor_linear1(x)
    action_probs = self._actor_relu(action_probs)
    action_probs = self._actor_linear2(action_probs)
    action_probs = self._actor_softmax(action_probs)

    critic_value = self._critic_linear1(x)
    critic_value = self._critic_relu(critic_value)
    critic_value = self._critic_linear2(critic_value)

    return action_probs, critic_value
