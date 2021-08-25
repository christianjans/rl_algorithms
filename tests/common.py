from typing import List, Tuple

import gym


def run_agent_in_environment(agent,
                             env: gym.Env,
                             episodes: int,
                             return_goal: float) -> Tuple[bool, List[str]]:
  # NOTE: The return is the undiscounted return.
  info = []

  for episode_index in range(episodes):
    episode_return = 0.0
    done = False
    observation = env.reset()

    while not done:
      action = agent.act(observation, explore=True)
      next_observation, reward, done, _ = env.step(action)
      agent.step(observation, action, reward, next_observation, done)
      episode_return += reward
      observation = next_observation

    info.append(f"Return in episode #{episode_index + 1}: {episode_return}")

    if episode_return >= return_goal:
      return True, info

  info.append(
      f"Unable to reach return goal of {return_goal} in {episodes} episodes.")

  return False, info
