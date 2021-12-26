import dataclasses

import numpy as np

from agent import Agent


@dataclasses.dataclass(frozen=True)
class QLearningConfig:
 alpha: float
 gamma: float
 epsilon: float
 num_states: int
 num_actions: int
 initial_value: float = 0.0


class QLearning(Agent):
  """Tabular version of the Q Learning algorithm."""

  def __init__(self, config: QLearningConfig):
    self._alpha = config.alpha
    self._gamma = config.gamma
    self._epislon = config.epsilon
    self._num_actions = config.num_actions
    self._next_action = None
    self._q = [[config.initial_value for _ in range(config.num_actions)]
               for _ in range(config.num_states)]

  def act(self, state: int, explore=True) -> int:
    return self._choose_action(state)

  def step(self,
           state: int,
           action: int,
           reward: float,
           next_state: int,
           done: bool):
    q = self._q[state][action]
    max_q = 0.0 if done else np.max(self._q[next_state])
    self._q[state][action] = \
        q + self._alpha * (reward + self._gamma * max_q - q)

  def _choose_action(self, state: int) -> int:
    if np.random.uniform(0, 1) < self._epislon:
      # Choose an action randomly.
      action = np.random.randint(0, self._num_actions)
    else:
      # Choose an action greedily.
      action = np.argmax(self._q[state])
    return action
