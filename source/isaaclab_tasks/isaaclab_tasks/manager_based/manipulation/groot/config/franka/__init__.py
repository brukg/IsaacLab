# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Groot-Lift-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.groot_lift_env_cfg:FrankaGr00tLiftEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Groot-Lift-Cube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.groot_lift_env_cfg:FrankaGr00tLiftEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
