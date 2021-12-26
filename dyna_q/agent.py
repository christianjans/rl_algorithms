import dataclasses

import numpy as np
import random

from agent import Agent


@dataclasses.dataclass(frozen=True)
class DynaQConfig:
  alpha: float
  gamma: float
  epsilon: float
  num_states: int
  num_actions: int
  planning_iterations: int
  initial_value: float = 0.0


class DynaQ(Agent):
  def __init__(self, config: DynaQConfig):
    self._alpha = config.alpha
    self._gamma = config.gamma
    self._epsilon = config.epsilon
    self._num_actions = config.num_actions
    self._num_planning_iterations = config.planning_iterations
    self._q = [[config.initial_value for _ in range(config.num_actions)]
               for _ in range(config.num_states)]
    self._model = {}

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

    # Update the model and do some planning.
    self._model[(state, action)] = (reward, next_state, done)
    self._q_plan()

  def _choose_action(self, state: int) -> int:
    if np.random.uniform(0, 1) < self._epsilon:
      action = np.random.randint(0, self._num_actions)
    else:
      action = np.argmax(self._q[state])
    return action

  def _q_plan(self):
    if (len(self._model) == 0):
      # Do not plan if we don't have any experience in the model.
      return

    import random
    for _ in range(self._num_planning_iterations):
      state, action = random.choice(list(self._model.keys()))
      reward, next_state, done = self._model[(state, action)]
      q = self._q[state][action]
      max_q = 0.0 if done else np.max(self._q[next_state])
      self._q[state][action] = \
          q + self._alpha * (reward + self._gamma * max_q - q)

  # def _update_q(self, state, action, reward, next_state, done):
  #   q = self._q[state][action]
  #   max_q = 0.0 if done else max(self._q[next_state])
  #   self._q[state][action] = \
  #       q + self._alpha * (reward + self._gamma * max_q - q)
