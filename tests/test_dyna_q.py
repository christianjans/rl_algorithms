import unittest

import gym

from dyna_q.agent import DynaQ, DynaQConfig
from tests.common import run_agent_in_environment


class SarsaTest(unittest.TestCase):
  def test_dyna_q_cliff_walking_v0(self):
    env = gym.make("CliffWalking-v0")

    EPISODES_ALLOWED = 500
    TARGET_RETURN = -30.0  # Finish the 4x12 grid in 30 time steps or less.

    sarsa_config = DynaQConfig(alpha=0.01,
                               gamma=0.9,
                               epsilon=0.1,
                               num_states=env.observation_space.n,
                               num_actions=env.action_space.n,
                               planning_iterations=50,
                               initial_value=0.0)
    sarsa_agent = DynaQ(sarsa_config)

    completed, info = run_agent_in_environment(
        sarsa_agent, env, EPISODES_ALLOWED, TARGET_RETURN)

    print(info)

    self.assertTrue(completed, msg="\n".join(info))
