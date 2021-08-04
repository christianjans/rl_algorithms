import collections
import random
from typing import List


class ReplayBuffer:
  def __init__(self, size: int):
    self._buffer = collections.deque([], maxlen=size)

  def add(self, transition):
    self._buffer.append(transition)

  def sample(self, batch_size: int) -> List:
    return random.sample(self._buffer, batch_size)

  def __len__(self) -> int:
    return len(self._buffer)
