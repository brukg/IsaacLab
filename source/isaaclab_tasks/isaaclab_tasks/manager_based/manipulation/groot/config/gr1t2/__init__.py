# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Groot-PickPlace-GR1T2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.groot_pickplace_env_cfg:GR1T2Gr00tPickPlaceEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Groot-PickPlace-GR1T2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.groot_pickplace_env_cfg:GR1T2Gr00tPickPlaceEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
