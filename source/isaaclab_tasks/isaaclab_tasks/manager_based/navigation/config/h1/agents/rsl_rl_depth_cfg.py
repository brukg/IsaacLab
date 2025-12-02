# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RSL-RL Depth CNN for H1 navigation - DEPTH CAMERA ONLY."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlActorCriticDepthCNNCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class H1NavigationDepthCNNRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for H1 navigation with DEPTH CAMERA ONLY."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "h1_navigation_depth"

    # Observation groups: actor uses proprio+depth, critic uses proprio+depth
    obs_groups = {
        "policy": ["proprio", "depth"],
        "critic": ["proprio", "depth"],
    }

    policy = RslRlActorCriticDepthCNNCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # Proprio: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + pose_command(4) = 13
        num_actor_obs_prop=13,
        # Depth camera: 53×30 = 1590 pixels
        obs_depth_shape=(53, 30),
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
