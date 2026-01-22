# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR1T2 humanoid environment configuration for Isaac-Gr00t VLA inference.

This extends the existing NutPourGR1T2PinkIKEnvCfg which already has cameras,
and adds the specific observation terms that Gr00t expects for the GR1 embodiment.

Gr00t GR1 embodiment expects:
- Video: 256x256 RGB image (ego_view_bg_crop_pad_res256_freq20)
- State: left_arm (7), right_arm (7), left_hand (6), right_hand (6), waist (3) = 29 total
- Action: left_arm (7), right_arm (7), left_hand (6), right_hand (6), waist (3) = 29 total

IMPORTANT: Action Representation
- left_arm, right_arm, left_hand, right_hand use RELATIVE actions (deltas from current state)
- waist uses ABSOLUTE actions (target positions)
- The GR00T server handles the relative-to-absolute conversion internally using the state observations

Note: Gr00t outputs absolute joint positions after internal conversion, so we use
JointPositionActionCfg instead of Pink IK which expects EEF poses.
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place import mdp
from isaaclab_tasks.manager_based.manipulation.pick_place.nutpour_gr1t2_pink_ik_env_cfg import (
    NutPourGR1T2PinkIKEnvCfg,
)


# GR1T2 joint name patterns for each body group
# Gr00t GR1 expects: left_arm (7), right_arm (7), left_hand (6), right_hand (6), waist (3)
GR1T2_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
]

GR1T2_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]

# Fourier hand has 6 DOF per hand (proximal joints for all 5 fingers + thumb pitch)
# IMPORTANT: The RoboCasa training data uses REVERSED hand joint order!
# See robocasa/models/robots/__init__.py:127 - "Reverse the order to match the real robot"
# The order below is REVERSED to match the training data convention.
GR1T2_LEFT_HAND_JOINTS = [
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
]

GR1T2_RIGHT_HAND_JOINTS = [
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
]

GR1T2_WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]

# Flag to control whether waist action from GR00T is applied
# If False, waist is held at current position (ignoring GR00T output)
APPLY_WAIST_ACTION = False

# All controllable joints in Gr00t action order
# Total: 7 + 7 + 6 + 6 + 3 = 29 joints
# NOTE: Waist must always be included to maintain position control
GR1T2_ALL_GROOT_JOINTS = (
    GR1T2_LEFT_ARM_JOINTS + GR1T2_RIGHT_ARM_JOINTS + GR1T2_LEFT_HAND_JOINTS + GR1T2_RIGHT_HAND_JOINTS + GR1T2_WAIST_JOINTS
)


@configclass
class GR1T2Gr00tPickPlaceEnvCfg(NutPourGR1T2PinkIKEnvCfg):
    """GR1T2 humanoid environment for Isaac-Gr00t VLA inference.

    Provides observations in the format expected by Gr00t GR1 embodiment.
    Uses joint position control (not Pink IK) since Gr00t outputs absolute joint positions.
    """

    # Gr00t client configuration - IsaacLab keys
    camera_keys: list[str] = ["robot_pov_cam"]

    # State keys - model REQUIRES waist state as input (always include)
    state_keys: list[str] = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
    groot_state_keys: list[str] = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]

    # Action keys - always include waist (needed to maintain position)
    action_keys: list[str] = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand", "action.waist"]

    # Flag to ignore waist action (use current position instead)
    apply_waist_action: bool = APPLY_WAIST_ACTION

    # Gr00t model expected keys (must match training config)
    groot_video_keys: list[str] = ["ego_view_bg_crop_pad_res256_freq20"]

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Set high stiffness for waist to hold position when not applying waist action
        # This prevents the waist from drifting due to arm movements
        self.scene.robot.actuators["trunk"] = ImplicitActuatorCfg(
            joint_names_expr=["waist_.*"],
            effort_limit=None,
            velocity_limit=None,
            stiffness=10000.0,  # Very high stiffness to hold position
            damping=100.0,  # High damping to prevent oscillation
            armature=0.01,
        )

        # Override the Pink IK action with joint position action for Gr00t
        # GR00T server converts RELATIVE actions to ABSOLUTE using the current state
        # The output after conversion is absolute joint positions (radians)
        self.actions.gr1_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(GR1T2_ALL_GROOT_JOINTS),
            scale=1.0,  # Actions are already in radians, no scaling needed
            use_default_offset=False,  # Don't use default offsets
        )

        # Add Gr00t-specific state observations for each body group
        self.observations.policy.left_arm = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": GR1T2_LEFT_ARM_JOINTS},
        )

        self.observations.policy.right_arm = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": GR1T2_RIGHT_ARM_JOINTS},
        )

        self.observations.policy.left_hand = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": GR1T2_LEFT_HAND_JOINTS},
        )

        self.observations.policy.right_hand = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": GR1T2_RIGHT_HAND_JOINTS},
        )

        # Always include waist observation - model REQUIRES it as input
        self.observations.policy.waist = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": GR1T2_WAIST_JOINTS},
        )


@configclass
class GR1T2Gr00tPickPlaceEnvCfg_PLAY(GR1T2Gr00tPickPlaceEnvCfg):
    """Play configuration with more environments."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
