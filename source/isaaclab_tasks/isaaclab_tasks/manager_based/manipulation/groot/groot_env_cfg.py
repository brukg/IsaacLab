# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base environment configuration for Isaac-Gr00t VLA policy inference.

This module provides base configurations for environments that work with
the Isaac-Gr00t inference server, including camera sensors and observation
groups formatted for VLA models.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.envs.mdp as mdp


##
# Scene definition with cameras
##


@configclass
class Gr00tSceneCfg(InteractiveSceneCfg):
    """Base scene configuration with camera sensors for Gr00t VLA inference.

    This scene includes:
    - A wrist-mounted camera on the robot end-effector
    - A scene camera providing an external view
    - Ground plane and lighting
    """

    # Robot: will be set by derived classes
    robot: ArticulationCfg = MISSING

    # Object: will be set by derived classes
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Wrist camera: will be attached to robot end-effector
    # Derived classes should override prim_path to match their robot structure
    wrist_camera: TiledCameraCfg = MISSING

    # Scene camera: external view of the workspace
    scene_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/SceneCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.5, 0.0, 1.0),
            rot=(0.9239, 0.0, 0.3827, 0.0),  # Looking at workspace
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


##
# Observation specifications for Gr00t
##


@configclass
class Gr00tObservationsCfg:
    """Observation configuration for Gr00t VLA inference.

    Observations are returned as a dictionary (concatenate_terms=False) to match
    the format expected by the Gr00t inference server.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations including cameras and robot state."""

        # Camera observations
        wrist_camera = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "data_type": "rgb"},
        )

        scene_camera = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("scene_camera"), "data_type": "rgb"},
        )

        # Robot state observations
        joint_positions = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        joint_velocities = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # Return as dict for Gr00t

    policy: PolicyCfg = PolicyCfg()


##
# Event specifications
##


@configclass
class Gr00tEventsCfg:
    """Event configuration for resetting the scene."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


##
# Termination specifications
##


@configclass
class Gr00tTerminationsCfg:
    """Termination conditions for Gr00t environments."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


##
# Base environment configuration
##


@configclass
class Gr00tEnvCfg(ManagerBasedRLEnvCfg):
    """Base environment configuration for Isaac-Gr00t VLA inference.

    This configuration:
    - Provides camera sensors for vision input
    - Returns observations as dictionaries (not concatenated)
    - Is designed for inference, not training

    Derived classes should:
    - Set the robot and object configurations
    - Configure the wrist camera attachment point
    - Set appropriate action configurations
    """

    # Scene settings
    scene: Gr00tSceneCfg = MISSING

    # Observation settings (dict format for Gr00t)
    observations: Gr00tObservationsCfg = Gr00tObservationsCfg()

    # Action settings: will be set by derived classes
    actions = MISSING

    # Event settings
    events: Gr00tEventsCfg = Gr00tEventsCfg()

    # Termination settings
    terminations: Gr00tTerminationsCfg = Gr00tTerminationsCfg()

    # No rewards/curriculum for inference
    rewards = None
    curriculum = None
    commands = None

    # List of camera observation keys for the Gr00t client
    camera_keys: list[str] = ["wrist_camera", "scene_camera"]

    # List of state observation keys for the Gr00t client
    state_keys: list[str] = ["joint_positions"]

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 30.0

        # Simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation

        # Disable recorders for inference
        self.recorders = None
