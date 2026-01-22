# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka lift environment configuration for Isaac-Gr00t VLA inference.

This extends the existing FrankaCubeLiftEnvCfg by adding camera observations
for VLA policy inference.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

# Import the existing working environment
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)


##
# Observation Configuration with cameras
##


@configclass
class FrankaGr00tObservationsCfg:
    """Observation configuration for Franka Gr00t inference with cameras."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations for Franka with cameras."""

        # Camera observations
        wrist_camera = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "data_type": "rgb"},
        )

        scene_camera = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("scene_camera"), "data_type": "rgb"},
        )

        # Robot state
        joint_positions = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # End-effector pose (position + quaternion combined)
        eef_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # Return as dict for Gr00t

    policy: PolicyCfg = PolicyCfg()


##
# Environment Configuration
##


@configclass
class FrankaGr00tLiftEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka lift cube environment for Isaac-Gr00t VLA inference.

    This extends the existing FrankaCubeLiftEnvCfg by adding:
    - Wrist camera on the panda hand
    - Scene camera for third-person view
    - Observations formatted for Gr00t (concatenate_terms=False)
    """

    # Override observations for Gr00t
    observations: FrankaGr00tObservationsCfg = FrankaGr00tObservationsCfg()

    # Gr00t client configuration
    camera_keys: list[str] = ["wrist_camera", "scene_camera"]
    state_keys: list[str] = ["joint_positions", "eef_pose"]
    action_keys: list[str] = ["arm_action", "gripper_action"]

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Add wrist camera on panda hand
        self.scene.wrist_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.04),
                rot=(0.7071, 0.0, 0.7071, 0.0),
                convention="ros",
            ),
            spawn=PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
            ),
            width=224,
            height=224,
            data_types=["rgb"],
        )

        # Add scene camera - third-person view
        self.scene.scene_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/SceneCamera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.2, 0.5, 0.8),
                rot=(0.9239, 0.0, 0.3827, 0.0),
                convention="ros",
            ),
            spawn=PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
            ),
            width=224,
            height=224,
            data_types=["rgb"],
        )

        # Disable recorders for inference
        self.recorders = None


@configclass
class FrankaGr00tLiftEnvCfg_PLAY(FrankaGr00tLiftEnvCfg):
    """Play configuration with more environments."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
