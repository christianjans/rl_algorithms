import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class MLPModelConfig:
  input_size: int
  output_size: int
  depth: int
  width: int


class MLPModel(torch.nn.Module):
  def __init__(self, config: MLPModelConfig):
    super(MLPModel, self).__init__()

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

  def forward(self, x: torch.Tensor):
    for layer in self._layers:
      x = layer(x)
    return x


if __name__ == "__main__":
  import numpy as np

  DATA_SIZE = 10
  EPOCHS = 100
  INPUT_SIZE = 5
  OUTPUT_SIZE = 2
  DEPTH = 4
  WIDTH = 16

  X = np.random.uniform(low=-1.0, high=1.0, size=(DATA_SIZE, INPUT_SIZE))
  X = torch.from_numpy(X.astype(np.float32))
  Y = np.random.uniform(low=-10.0, high=10.0, size=(DATA_SIZE, OUTPUT_SIZE))
  Y = torch.from_numpy(Y.astype(np.float32))

  config = MLPModelConfig(input_size=INPUT_SIZE,
                          output_size=OUTPUT_SIZE,
                          depth=DEPTH,
                          width=WIDTH)
  model = MLPModel(config)
  loss_function = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  for _ in range(EPOCHS):
    predictions = model(X)
    loss = loss_function(predictions, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss}")
