from typing import Sequence

import numpy as np


class Bandit:
  def __init__(self, k: int, mu: float = 0.0, sigma: float = 1.0):
    self._k = k
    self._mu = mu
    self._sigma = sigma
    self._actions = [action for action in range(k)]
    self._q_star = [np.random.normal(loc=mu, scale=sigma) for _ in range(k)]

  @property
  def k(self) -> int:
    return self._k

  @property
  def actions(self) -> Sequence[int]:
    return self._actions

  def step(self, action: int) -> float:
    if action < 0 or action >= self._k:
      raise ValueError(f"Invalid action: {action}.")

    reward = np.random.normal(loc=self._q_star[action], scale=self._sigma)
    return reward
