# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 GR00T environment configurations."""

import gymnasium as gym

from .groot_g1_env_cfg import G1Gr00tEnvCfg, G1Gr00tEnvCfg_PLAY

gym.register(
    id="Isaac-Groot-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": G1Gr00tEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Groot-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": G1Gr00tEnvCfg_PLAY},
    disable_env_checker=True,
)
