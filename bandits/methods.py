import dataclasses
from typing import Callable

import numpy as np


@dataclasses.dataclass
class MethodConfig:
  k: int = 10
  alpha_function: Callable[[int], float] = lambda n: 1 / n
  initial_value: float = 0.0


@dataclasses.dataclass
class ActionValueMethodConfig(MethodConfig):
  epsilon: float = 0.1


@dataclasses.dataclass
class UcbActionValueMethodConfig(MethodConfig):
  c: float = 2.0


@dataclasses.dataclass
class GradientMethodConfig(MethodConfig):
  pass


class Method:
  def act(self):
    raise NotImplementedError

  def step(self, action: int, reward: float):
    raise NotImplementedError


class ActionValueMethod(Method):
  def __init__(self, config: ActionValueMethodConfig):
    self._k = config.k
    self._epsilon = config.epsilon
    self._alpha_function = config.alpha_function
    self._q = np.array(
        [config.initial_value for _ in range(config.k)], dtype=np.float32)
    self._n = np.array([1 for _ in range(config.k)], dtype=np.int)

  def act(self) -> int:
    if self._epsilon > np.random.random():
      return np.random.randint(low=0, high=self._k)
    return np.argmax(self._q)

  def step(self, action: int, reward: float):
    current_alpha = self._alpha_function(self._n[action])
    old_estimate = self._q[action]

    self._q[action] = old_estimate + current_alpha * (reward - old_estimate)

    self._n[action] += 1


class UcbActionValueMethod(Method):
  def __init__(self, config: UcbActionValueMethodConfig):
    self._k = config.k
    self._c = config.c
    self._alpha_function = config.alpha_function
    self._q = np.array(
        [config.initial_value for _ in range(config.k)], dtype=np.float32)
    self._n = np.array([1 for _ in range(config.k)], dtype=np.int32)
    self._t = 1

  def act(self):
    action = np.argmax(self._ucb)
    return action

  def step(self, action: int, reward: float):
    current_alpha = self._alpha_function(self._n[action])
    old_estimate = self._q[action]

    self._q[action] = old_estimate + current_alpha * (reward - old_estimate)

    self._n[action] += 1
    self._t += 1

  @property
  def _ucb(self):
    return self._q + self._c * np.sqrt(np.log(self._t) / self._n)


class GradientMethod(Method):
  def __init__(self, config: GradientMethodConfig):
    self._k = config.k
    self._alpha_function = config.alpha_function
    self._H = np.array(
        [config.initial_value for _ in range(config.k)], dtype=np.float32)
    self._n = np.array([1 for _ in range(config.k)], dtype=np.int32)
    self._mean_reward = 0.0
    self._t = 1

  def act(self):
    action = np.random.choice(np.arange(0, self._k), p=self._pi)
    return action

  def step(self, action: int, reward: float):
    alpha = self._alpha_function(self._n[action])
    mean_reward = self._mean_reward
    pi = self._pi

    for current_action in range(self._k):
      old_estimate = self._H[current_action]
      if current_action == action:
        self._H[current_action] = old_estimate + \
            alpha * (reward - mean_reward) * (1 - pi[current_action])
      else:
        self._H[current_action] = old_estimate - \
            alpha * (reward - mean_reward) * pi[current_action]

    self._mean_reward += (reward - mean_reward) / self._t
    self._n[action] += 1
    self._t += 1

  @property
  def _pi(self):
    return np.exp(self._H) / np.sum(np.exp(self._H))
