# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 navigation environment configurations."""

import gymnasium as gym

from . import agents

##
# Register Gym environments - Base Navigation
##

gym.register(
    id="Isaac-Navigation-Flat-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1NavigationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Flat-H1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1NavigationPPORunnerCfg",
    },
)

##
# Vision Environments - Depth Camera Only
##

gym.register(
    id="Isaac-Navigation-Flat-H1-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_cfg:H1NavigationDepthCNNRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Flat-H1-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationDepthEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_cfg:H1NavigationDepthCNNRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Flat-H1-Depth-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_rnn_cfg:H1NavigationDepthCNNRNNRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Flat-H1-Depth-RNN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationDepthEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_rnn_cfg:H1NavigationDepthCNNRNNRunnerCfg",
    },
)

##
# Vision Environments - Height Scanner Only
##

gym.register(
    id="Isaac-Navigation-Flat-H1-Scanner-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationScannerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_scanner_cfg:H1NavigationScannerRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Flat-H1-Scanner-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationScannerEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_scanner_cfg:H1NavigationScannerRunnerCfg",
    },
)

##
# Vision Environments - Depth + Height Scanner Combined
##

gym.register(
    id="Isaac-Navigation-Flat-H1-DepthScanner-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationDepthScannerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_scanner_cfg:H1NavigationDepthScannerRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Flat-H1-DepthScanner-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationDepthScannerEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_scanner_cfg:H1NavigationDepthScannerRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Flat-H1-DepthScanner-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationDepthScannerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_scanner_rnn_cfg:H1NavigationDepthScannerRNNRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Flat-H1-DepthScanner-RNN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationDepthScannerEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_scanner_rnn_cfg:H1NavigationDepthScannerRNNRunnerCfg",
    },
)


##
# Register Gym environments - Rough Terrain Navigation
##

gym.register(
    id="Isaac-Navigation-Rough-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1NavigationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Rough-H1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1NavigationPPORunnerCfg",
    },
)

##
# Rough Terrain - Vision (Depth Camera)
##

gym.register(
    id="Isaac-Navigation-Rough-H1-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationRoughVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_cfg:H1NavigationDepthCNNRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Rough-H1-Vision-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationRoughVisionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_cfg:H1NavigationDepthCNNRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Rough-H1-Vision-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationRoughVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_rnn_cfg:H1NavigationDepthCNNRNNRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Rough-H1-Vision-RNN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationRoughVisionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_rnn_cfg:H1NavigationDepthCNNRNNRunnerCfg",
    },
)

##
# Rough Terrain - Vision + Scanner Combined
##

gym.register(
    id="Isaac-Navigation-Rough-H1-VisionScanner-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationRoughVisionScannerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_scanner_cfg:H1NavigationDepthScannerRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Rough-H1-VisionScanner-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationRoughVisionScannerEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_scanner_cfg:H1NavigationDepthScannerRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Rough-H1-VisionScanner-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationRoughVisionScannerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_scanner_rnn_cfg:H1NavigationDepthScannerRNNRunnerCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Rough-H1-VisionScanner-RNN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:H1NavigationRoughVisionScannerEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_depth_scanner_rnn_cfg:H1NavigationDepthScannerRNNRunnerCfg",
    },
)