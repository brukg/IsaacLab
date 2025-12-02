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
class G1RoughTD3RunnerCfg(RslRlOffPolicyRunnerCfg):
    """TD3 configuration for Unitree G1 humanoid on rough terrain."""

    num_steps_per_env = 1
    max_iterations = 3000
    save_interval = 500
    experiment_name = "g1_rough_td3"

    random_steps = 5000
    gradient_steps = 20

    policy = RslRlTd3ActorCriticCfg(
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlTd3AlgorithmCfg(
        learning_rate_actor=1e-4,
        learning_rate_critic=1e-4,
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
class G1FlatTD3RunnerCfg(RslRlOffPolicyRunnerCfg):
    """TD3 configuration for Unitree G1 humanoid on flat terrain."""

    num_steps_per_env = 1
    max_iterations = 3000
    save_interval = 500
    experiment_name = "g1_flat_td3"

    random_steps = 5000
    gradient_steps = 20

    policy = RslRlTd3ActorCriticCfg(
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlTd3AlgorithmCfg(
        learning_rate_actor=1e-4,
        learning_rate_critic=1e-4,
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
class G1RoughFastTD3RunnerCfg(RslRlOffPolicyRunnerCfg):
    """FastTD3 configuration for Unitree G1 humanoid on rough terrain."""

    num_steps_per_env = 1
    max_iterations = 3000
    save_interval = 500
    experiment_name = "g1_rough_fast_td3"

    random_steps = 5000
    gradient_steps = 20

    policy = RslRlTd3ActorCriticCfg(
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlFastTd3AlgorithmCfg(
        learning_rate_actor=1e-4,
        learning_rate_critic=1e-4,
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


@configclass
class G1FlatFastTD3RunnerCfg(RslRlOffPolicyRunnerCfg):
    """FastTD3 configuration for Unitree G1 humanoid on flat terrain."""

    num_steps_per_env = 1
    max_iterations = 3000
    save_interval = 500
    experiment_name = "g1_flat_fast_td3"

    random_steps = 5000
    gradient_steps = 20

    policy = RslRlTd3ActorCriticCfg(
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlFastTd3AlgorithmCfg(
        learning_rate_actor=1e-4,
        learning_rate_critic=1e-4,
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
