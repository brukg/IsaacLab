# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RSL-RL MLP for H1 navigation - HEIGHT SCANNER ONLY."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class H1NavigationScannerRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for H1 navigation with HEIGHT SCANNER ONLY."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "h1_navigation_scanner"

    # Observation groups: actor uses proprio+scanner, critic uses proprio+scanner
    obs_groups = {
        "policy": ["proprio", "scanner"],
        "critic": ["proprio", "scanner"],
    }

    # MLP policy (scanner is flattened, so MLP works)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
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
