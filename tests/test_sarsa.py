import unittest

import gym

from sarsa.agent import Sarsa, SarsaConfig
from tests.common import run_agent_in_environment


class SarsaTest(unittest.TestCase):
  def test_sarsa_cliff_walking_v0(self):
    env = gym.make("CliffWalking-v0")

    EPISODES_ALLOWED = 500
    TARGET_RETURN = -30.0  # Finish the 4x12 grid in 30 time steps or less.

    sarsa_config = SarsaConfig(alpha=0.01,
                               gamma=0.9,
                               epsilon=0.1,
                               num_states=env.observation_space.n,
                               num_actions=env.action_space.n,
                               initial_value=0.0)
    sarsa_agent = Sarsa(sarsa_config)

    completed, info = run_agent_in_environment(
        sarsa_agent, env, EPISODES_ALLOWED, TARGET_RETURN)

    self.assertTrue(completed, msg="\n".join(info))
