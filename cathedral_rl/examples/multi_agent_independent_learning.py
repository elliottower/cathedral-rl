# flake8: noqa

import argparse

import ray
from pettingzoo.sisl import waterworld_v4
from ray import air, tune
from ray.rllib.algorithms.apex_ddpg import ApexDDPGConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

parser = argparse.ArgumentParser()
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--num-gpus",
    type=int,
    default=0,
    help="Number of GPUs to use for training.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: Only one episode will be "
    "sampled.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    def env_creator(args):
        return PettingZooEnv(waterworld_v4.env())

    env = env_creator({})
    register_env("waterworld", env_creator)

    config = (
        ApexDDPGConfig()
        .environment("waterworld")
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=2)
        .multi_agent(
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )

    if args.as_test:
        # Only a compilation test of running waterworld / independent learning.
        stop = {"training_iteration": 1}
    else:
        stop = {"episodes_total": 60000}

    tune.Tuner(
        "APEX_DDPG",
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space=config,
    ).fit()
