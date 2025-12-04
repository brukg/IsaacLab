# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Carter differential drive robot locomotion on flat terrain."""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    MySceneCfg,
    CommandsCfg,
    ActionsCfg,
    EventCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.carter import CARTER_V1_CFG  # isort: skip


@configclass
class CarterFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Carter differential drive robot locomotion on flat terrain."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------
        # Scene settings
        # ------------------------------
        self.scene.robot = CARTER_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Remove height scanner (not needed for wheeled robot on flat terrain)
        self.scene.height_scanner = None

        # Update contact sensor for wheels
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"

        # ------------------------------
        # MDP settings
        # ------------------------------

        # Commands - disable y-velocity for differential drive
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # No lateral movement
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # Actions - use velocity control for wheels
        self.actions.joint_pos = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["left_wheel", "right_wheel"],
            scale=5.0,  # Reduced from 10.0 for more stable control
        )

        # Observations - adapted for wheeled robot
        @configclass
        class CarterObservationsCfg:
            """Observation specifications for Carter robot."""

            @configclass
            class PolicyCfg(ObsGroup):
                """Observations for policy group."""

                # observation terms (order preserved)
                base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
                base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
                projected_gravity = ObsTerm(
                    func=mdp.projected_gravity,
                    noise=Unoise(n_min=-0.05, n_max=0.05),
                )
                velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
                joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
                actions = ObsTerm(func=mdp.last_action)

                def __post_init__(self):
                    self.enable_corruption = True
                    self.concatenate_terms = True

            # observation groups
            policy: PolicyCfg = PolicyCfg()

        self.observations = CarterObservationsCfg()

        # ------------------------------
        # Rewards
        # ------------------------------

        # Tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Penalize z-axis linear velocity (should stay on ground)
        self.rewards.lin_vel_z_l2.weight = -2.0

        # Penalize roll and pitch angular velocity (should stay upright)
        self.rewards.ang_vel_xy_l2.weight = -0.5

        # Penalize high torques
        self.rewards.dof_torques_l2.weight = -1.0e-4

        # Penalize high accelerations
        self.rewards.dof_acc_l2.weight = -1.0e-6

        # Penalize high action rates
        self.rewards.action_rate_l2.weight = -0.005

        # Remove foot-related rewards (not applicable for wheeled robots)
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits = None

        # ------------------------------
        # Terminations
        # ------------------------------

        # Terminate if robot tips over (base contacts ground)
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

        # Reduce mass randomization range
        self.events.add_base_mass.params["asset_cfg"].body_names = "chassis_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)

        # Disable COM randomization
        self.events.base_com = None

        # Keep push events for robustness
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}

        # ------------------------------
        # Curriculum
        # ------------------------------

        # Disable terrain curriculum (flat terrain)
        self.curriculum.terrain_levels = None


@configclass
class CarterFlatEnvCfg_PLAY(CarterFlatEnvCfg):
    """Configuration for Carter robot locomotion on flat terrain for evaluation."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
