# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Carter differential drive robot locomotion on rough terrain."""

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.carter import CARTER_V1_CFG  # isort: skip


@configclass
class CarterRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Carter differential drive robot locomotion on rough terrain."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------
        # Scene settings
        # ------------------------------
        self.scene.robot = CARTER_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Keep rough terrain settings from parent
        # Scale down the terrains significantly for wheeled robot (much gentler than legged robots)
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.04)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.03)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Disable very difficult terrains for wheeled robots
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.02, 0.04)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.02, 0.04)

        # Update contact sensor for wheels
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"

        # Remove height scanner (wheeled robots typically don't need it)
        # If you want to keep it for terrain awareness, uncomment the line below
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/chassis_link"
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # ------------------------------
        # MDP settings
        # ------------------------------

        # Commands - disable y-velocity for differential drive
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)  # Slower on rough terrain
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # No lateral movement
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)  # Slower rotation
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # Actions - use velocity control for wheels
        self.actions.joint_pos = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel", "right_wheel"],
            scale=5.0,  # Reduced from 10.0 for more stable control
        )

        # ------------------------------
        # Rewards
        # ------------------------------

        # Tracking rewards - slightly higher than flat to encourage progress on rough terrain
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.2

        # Penalize z-axis linear velocity (should stay on ground) - slightly relaxed from flat
        self.rewards.lin_vel_z_l2.weight = -1.5

        # Penalize roll and pitch angular velocity - relaxed from flat (rough terrain causes natural tilt)
        self.rewards.ang_vel_xy_l2.weight = -0.3

        # Penalize high torques - same as flat
        self.rewards.dof_torques_l2.weight = -1.0e-4

        # Penalize high accelerations - same as flat
        self.rewards.dof_acc_l2.weight = -1.0e-6

        # Penalize high action rates - same as flat
        self.rewards.action_rate_l2.weight = -0.005

        # Keep robot upright - slightly relaxed from flat (rough terrain causes natural tilt)
        self.rewards.flat_orientation_l2.weight = -0.75

        # Remove foot-related rewards (not applicable for wheeled robots)
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.dof_pos_limits = None

        # ------------------------------
        # Terminations
        # ------------------------------

        # Terminate if robot tips over (chassis contacts ground)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "chassis_link"

        # ------------------------------
        # Events
        # ------------------------------

        # Disable joint reset events (wheeled robot has no joint positions to reset)
        self.events.reset_robot_joints = None

        # Simplify base reset
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Disable physics material randomization
        self.events.physics_material = None

        # Mass randomization - match flat terrain (not more aggressive)
        self.events.add_base_mass.params["asset_cfg"].body_names = "chassis_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)

        # Disable COM randomization
        self.events.base_com = None

        # Push events - match flat terrain magnitude
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}

        # Disable external forces - too disruptive for wheeled robot on rough terrain
        # The terrain geometry itself provides sufficient challenge
        self.events.base_external_force_torque = None


@configclass
class CarterRoughEnvCfg_PLAY(CarterRoughEnvCfg):
    """Configuration for Carter robot locomotion on rough terrain for evaluation."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
