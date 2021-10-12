import dataclasses
import datetime
import os
from typing import Any, ClassVar, Dict

import matplotlib.pyplot as plt
import numpy as np

from bandits.bandit import Bandit
import bandits.methods as methods


@dataclasses.dataclass
class ExperimentConfig:
  # Parameters for the experiment.
  num_steps: int = 1000
  num_runs: int = 2000
  seed: int = 1
  save_directory: str = "bandits/data/"

  # Parameters for the bandit.
  k: int = 10
  mu: float = 0
  sigma: float = 1

  # Parameters for the method.
  method_class: ClassVar[methods.Method] = methods.GradientMethod
  method_config: methods.MethodConfig = methods.GradientMethodConfig(
      k=k, alpha_function=lambda _: 0.1, initial_value=0.0)


def run_experiment(config: ExperimentConfig):
  np.random.seed(config.seed)

  if not os.path.exists(config.save_directory):
      os.makedirs(config.save_directory)

  rewards_data = np.empty((config.num_steps, config.num_runs))

  for run in range(config.num_runs):
    bandit = Bandit(k=config.k, mu=config.mu, sigma=config.sigma)
    method = config.method_class(config.method_config)

    print(f"Starting run {run + 1}/{config.num_runs}")

    for time_step in range(config.num_steps):
      action = method.act()
      reward = bandit.step(action=action)
      method.step(action=action, reward=reward)
      rewards_data[time_step][run] = reward

  means = np.mean(rewards_data, axis=-1)
  time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
  plt.title(f"{config.method_class.__name__}: num_steps={config.num_steps}, "
            f"num_runs={config.num_runs}, seed={config.seed}, k={config.k}, "
            f"mu={config.mu}, sigma={config.sigma}\n{config.method_config}",
            fontdict={"fontsize": 6})
  plt.xlabel("Steps")
  plt.ylabel("Mean Reward")
  plt.plot(means)
  plt.savefig(os.path.join(
      config.save_directory,
      f"{time_string}-mean_reward-{config.method_class.__name__}.png"))


if __name__ == "__main__":
  config = ExperimentConfig()
  run_experiment(config=config)
