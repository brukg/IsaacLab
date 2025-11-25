# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlFastTd3AlgorithmCfg,
    RslRlOffPolicyRunnerCfg,
    RslRlTd3ActorCriticCfg,
    RslRlTd3AlgorithmCfg,
)


@configclass
class SpotFlatTD3RunnerCfg(RslRlOffPolicyRunnerCfg):
    """TD3 configuration for Boston Dynamics Spot-like quadruped on flat terrain."""

    num_steps_per_env = 1
    max_iterations = 20000
    save_interval = 500
    experiment_name = "spot_flat_td3"
    store_code_state = False

    random_steps = 10000
    gradient_steps = 1

    policy = RslRlTd3ActorCriticCfg(
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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


@configclass
class SpotFlatFastTD3RunnerCfg(RslRlOffPolicyRunnerCfg):
    """FastTD3 configuration for Boston Dynamics Spot-like quadruped on flat terrain."""

    num_steps_per_env = 1
    max_iterations = 16000
    save_interval = 500
    experiment_name = "spot_flat_fast_td3"
    store_code_state = False

    random_steps = 10000
    gradient_steps = 2

    policy = RslRlTd3ActorCriticCfg(
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlFastTd3AlgorithmCfg(
        learning_rate_actor=3e-4,
        learning_rate_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=1,
        max_grad_norm=1.0,
        replay_buffer_size=1000000,
        batch_size=256,
        exploration_noise=0.1,
        num_critic_updates=2,
    )
