# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Ranger Mini omni-wheel robot locomotion tasks."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-RangerMini-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.ranger_mini.flat_env_cfg:RangerMiniFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.ranger_mini.agents.rsl_rl_ppo_cfg:RangerMiniFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-RangerMini-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.ranger_mini.flat_env_cfg:RangerMiniFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.ranger_mini.agents.rsl_rl_ppo_cfg:RangerMiniFlatPPORunnerCfg",
    },
)
