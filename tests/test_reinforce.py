from tests.common import run_agent_in_environment
import unittest

import gym

from reinforce.agent import Reinforce, ReinforceConfig


class ReinforceTest(unittest.TestCase):
  def test_reinforce_cartpole_v0(self):
    env = gym.make("CartPole-v0")

    EPISODES_ALLOWED = 2000
    MAX_RETURN = 200.0

    reinforce_config = ReinforceConfig(nn_width=32,
                                       nn_depth=4,
                                       optimize_rate=10,
                                       learning_rate=0.001,
                                       gamma=0.999,
                                       device="cpu",
                                       observation_shape=env.observation_space.shape,
                                       action_size=env.action_space.n)
    reinforce_agent = Reinforce(reinforce_config)

    completed, info = run_agent_in_environment(
        reinforce_agent, env, EPISODES_ALLOWED, MAX_RETURN)

    self.assertTrue(completed, msg="\n".join(info))
