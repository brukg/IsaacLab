# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOffPolicyRunnerCfg, RslRlTd3ActorCriticCfg, RslRlTd3AlgorithmCfg


@configclass
class AntTD3RunnerCfg(RslRlOffPolicyRunnerCfg):
    num_steps_per_env = 1
    max_iterations = 10000
    save_interval = 200
    experiment_name = "ant_direct_td3"

    # Off-policy specific parameters
    random_steps = 10000
    gradient_steps = 1

    policy = RslRlTd3ActorCriticCfg(
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlTd3AlgorithmCfg(
        learning_rate_actor=3e-4,
        learning_rate_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        max_grad_norm=1.0,
        replay_buffer_size=1000000,
        batch_size=256,
        exploration_noise=0.1,
    )
