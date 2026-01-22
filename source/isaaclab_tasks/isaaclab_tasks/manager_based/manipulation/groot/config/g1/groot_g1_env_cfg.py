# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 environment configuration for Isaac-Gr00t VLA inference.

GR00T unitree_g1 embodiment expects:
- Video: ego_view (256x256)
- State: left_leg (6), right_leg (6), waist (3), left_arm (7), right_arm (7), left_hand (7), right_hand (7)
- Action: left_arm (7), right_arm (7), left_hand (7), right_hand (7), waist (3),
          base_height_command (1), navigate_command (2)

Action Representation (from embodiment_configs.py):
- left_arm, right_arm: RELATIVE (delta from current state)
- left_hand, right_hand: ABSOLUTE (G1 hand controlled like a gripper)
- waist: ABSOLUTE
- base_height_command, navigate_command: ABSOLUTE

For fixed-base manipulation, we ignore leg states/actions and locomotion commands.
The server handles relative-to-absolute conversion internally using state observations.
"""

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as base_mdp

from isaaclab_tasks.manager_based.manipulation.pick_place import mdp
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg import (
    PickPlaceG1InspireFTPEnvCfg,
    ObjectTableSceneCfg,
    ObservationsCfg as BaseObservationsCfg,
    TerminationsCfg,
    EventCfg,
)

# G1 joint configurations for GR00T
# Arms: 7 DOF each (shoulder pitch/roll/yaw, elbow, wrist yaw/roll/pitch)
G1_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
]

G1_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]

# Hands: Inspire hand - 7 controllable joints per hand
G1_LEFT_HAND_JOINTS = [
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_ring_proximal_joint",
    "L_pinky_proximal_joint",
    "L_thumb_intermediate_joint",
]

G1_RIGHT_HAND_JOINTS = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
    "R_thumb_intermediate_joint",
]

# Waist: 3 DOF
G1_WAIST_JOINTS = [
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
]

# Legs: 6 DOF each (for state observation only, not controlled in fixed-base)
G1_LEFT_LEG_JOINTS = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]

G1_RIGHT_LEG_JOINTS = [
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

# All controllable joints for action (arms + hands + waist = 7+7+7+7+3 = 31)
G1_ALL_GROOT_ACTION_JOINTS = (
    G1_LEFT_ARM_JOINTS + G1_RIGHT_ARM_JOINTS +
    G1_LEFT_HAND_JOINTS + G1_RIGHT_HAND_JOINTS +
    G1_WAIST_JOINTS
)


@configclass
class G1Gr00tActionsCfg:
    """Action configuration for G1 with GR00T - joint position control."""

    g1_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(G1_ALL_GROOT_ACTION_JOINTS),
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class G1Gr00tObservationsCfg:
    """Observation configuration for G1 with GR00T."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for GR00T policy."""

        # State observations in GR00T unitree_g1 order
        left_leg = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": G1_LEFT_LEG_JOINTS},
        )
        right_leg = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": G1_RIGHT_LEG_JOINTS},
        )
        waist = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": G1_WAIST_JOINTS},
        )
        left_arm = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": G1_LEFT_ARM_JOINTS},
        )
        right_arm = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": G1_RIGHT_ARM_JOINTS},
        )
        left_hand = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": G1_LEFT_HAND_JOINTS},
        )
        right_hand = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": G1_RIGHT_HAND_JOINTS},
        )

        # Camera observation
        ego_cam = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ego_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1Gr00tSceneCfg(ObjectTableSceneCfg):
    """Scene configuration for G1 with GR00T - adds camera to existing scene."""

    # Camera for ego view - GR00T expects 256x256 images
    ego_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/EgoCam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.1, 5.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.12, 1.67), rot=(-0.19848, 0.9801, 0.0, 0.0), convention="ros"),
    )


@configclass
class G1Gr00tEnvCfg(PickPlaceG1InspireFTPEnvCfg):
    """G1 environment configuration for GR00T inference.

    Extends the existing G1 pick-place environment with GR00T-specific observations
    and joint position actions (no Pink IK).
    """

    # Override scene to add camera
    scene: G1Gr00tSceneCfg = G1Gr00tSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # Override actions to use joint position control instead of Pink IK
    actions: G1Gr00tActionsCfg = G1Gr00tActionsCfg()

    # Override observations with GR00T-specific observations
    observations: G1Gr00tObservationsCfg = G1Gr00tObservationsCfg()

    # Keep terminations and events from parent
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # GR00T client configuration - IsaacLab observation keys
    camera_keys: list[str] = ["ego_cam"]
    # State keys in GR00T unitree_g1 order
    state_keys: list[str] = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
    # Action keys - excluding base_height_command and navigate_command for fixed-base
    action_keys: list[str] = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand", "action.waist"]

    # GR00T model keys (must match unitree_g1 embodiment)
    groot_video_keys: list[str] = ["ego_view"]
    groot_state_keys: list[str] = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]

    def __post_init__(self):
        """Post initialization."""
        # Skip parent's __post_init__ as it sets up Pink IK
        # Just set the basic simulation parameters
        self.decimation = 5
        self.episode_length_s = 30.0
        self.sim.dt = 1 / 100
        self.sim.render_interval = 2
        self.image_obs_list = ["ego_cam"]


@configclass
class G1Gr00tEnvCfg_PLAY(G1Gr00tEnvCfg):
    """Play configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
