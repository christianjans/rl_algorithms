class Agent:
  def act(self, observation, *args, **kwargs):
    raise NotImplementedError

  def step(self, *args, **kwargs):
    raise NotImplementedError
