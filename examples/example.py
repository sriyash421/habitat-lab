#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym

import habitat.gym  # noqa: F401
import argparse
# DEFAULT_CFG = "benchmark/rearrange/play/play.yaml"
DEFAULT_CFG="benchmark/rearrange/hab3_bench/single_agent_bench.yaml"
SENSOR_KEY="head_rgb"
from display_utils import display_rgb

def example():
    # Note: Use with for the example testing, doesn't need to be like this on the README
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    config = habitat.get_config(args.cfg, args.opts)
    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task
        env_config.max_episode_steps = 0
        
    # with gym.make("HabitatRearrangeEmpty-v0") as env:
    with habitat.Env(config=config) as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841
        print(observations.keys())
        display_rgb(observations[SENSOR_KEY])
        print("Agent acting inside environment.")
        count_steps = 0
        terminal = False
        while not terminal:
            observations, reward, terminal, info = env.step(
                env.action_space.sample()
            )  # noqa: F841
            count_steps += 1
            display_rgb(observations[SENSOR_KEY])
        print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    example()
