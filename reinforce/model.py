import dataclasses

import torch


@dataclasses.dataclass
class MLPDistributionModelConfig:
  input_size: int
  output_size: int
  depth: int
  width: int


class MLPDistributionModel(torch.nn.Module):
  def __init__(self, config: MLPDistributionModelConfig):
    super(MLPDistributionModel, self).__init__()

    self._layers = torch.nn.ModuleList()

    # Input layer.
    self._layers.append(torch.nn.Linear(config.input_size, config.width))
    self._layers.append(torch.nn.ReLU())

    # Torso layers.
    for _ in range(config.depth):
      self._layers.append(torch.nn.Linear(config.width, config.width))
      self._layers.append(torch.nn.ReLU())

    # Output layer.
    self._layers.append(torch.nn.Linear(config.width, config.output_size))
    self._layers.append(torch.nn.Softmax(dim=1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self._layers:
      x = layer(x)
    return x
    # print(x)
    # action = torch.multinomial(x, num_samples=1, replacement=True)
    # print(action)
    # action_log_prob = torch.log(x[0][action])
    # return action, action_log_prob
    # Always return the best action?
    # action = torch.argmax(x)
    # action_log_prob = torch.log(x[action])
    # return action, action_log_prob
