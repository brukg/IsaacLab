# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RSL-RL Depth CNN for H1 navigation - DEPTH + HEIGHT SCANNER."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlActorCriticDepthCNNCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class H1NavigationDepthScannerRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for H1 navigation with DEPTH + HEIGHT SCANNER."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "h1_navigation_depth_scanner"

    # Observation groups:
    # - Actor uses proprio+depth (for navigation)
    # - Critic uses proprio+depth+policy (policy includes height_scan for terrain awareness)
    obs_groups = {
        "policy": ["proprio", "depth"],
        "critic": ["proprio", "depth", "policy"],  # Includes height_scan from policy group
    }

    policy = RslRlActorCriticDepthCNNCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # CNN configs are set via __post_init__ in RslRlActorCriticDepthCNNCfg
        # Uses generic ActorCriticCNN with TensorDict observations
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
