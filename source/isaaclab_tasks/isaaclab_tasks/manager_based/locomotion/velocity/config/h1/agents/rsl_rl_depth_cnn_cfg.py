# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RSL-RL depth CNN policy for H1."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlActorCriticDepthCNNCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class H1RoughDepthCNNRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for H1 with depth camera using PPO."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "h1_rough_depth_cnn"

    # Observation groups mapping
    # Actor gets proprio first, then depth (as expected by ActorDepthCNN)
    # Critic uses proprio, depth, AND policy (which includes height_scan) for better terrain awareness
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
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
