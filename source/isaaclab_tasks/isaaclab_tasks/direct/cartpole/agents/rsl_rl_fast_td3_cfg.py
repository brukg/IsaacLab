# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlFastTd3AlgorithmCfg, RslRlOffPolicyRunnerCfg, RslRlTd3ActorCriticCfg


@configclass
class CartpoleFastTD3RunnerCfg(RslRlOffPolicyRunnerCfg):
    num_steps_per_env = 1
    max_iterations = 2500
    save_interval = 100
    experiment_name = "cartpole_direct_fast_td3"

    # Off-policy specific parameters
    random_steps = 500
    gradient_steps = 40  # More gradient steps for faster learning

    policy = RslRlTd3ActorCriticCfg(
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )

    algorithm = RslRlFastTd3AlgorithmCfg(
        learning_rate_actor=3e-4,
        learning_rate_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=1,  # Faster policy updates
        max_grad_norm=1.0,
        replay_buffer_size=100000,
        batch_size=256,
        exploration_noise=0.1,
        num_critic_updates=2,  # Multiple critic updates per step
    )
