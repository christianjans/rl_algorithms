import unittest

import gym

from dqn.agent import DQN, DQNConfig
from tests.common import run_agent_in_environment


class DQNTest(unittest.TestCase):
  def test_dqn_discrete_action(self):
    env = gym.make("CartPole-v0")

    EPISODES_ALLOWED = 200
    MAX_RETURN = 200.0

    dqn_config = DQNConfig(batch_size=128,
                           buffer_size=10000,
                           nn_width=16,
                           nn_depth=4,
                           optimize_rate=1,
                           target_update_rate=10,
                           learning_rate=0.001,
                           gamma=0.999,
                           epsilon_start=0.9,
                           epsilon_end=0.05,
                           epsilon_decay=200,
                           device="cpu",
                           observation_shape=env.observation_space.shape,
                           action_size=env.action_space.n)
    dqn_agent = DQN(dqn_config)

    completed, info = run_agent_in_environment(
        dqn_agent, env, EPISODES_ALLOWED, MAX_RETURN)

    self.assertTrue(completed, msg="\n".join(info))
