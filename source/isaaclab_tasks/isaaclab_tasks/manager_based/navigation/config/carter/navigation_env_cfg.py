# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Carter differential drive robot navigation on flat terrain."""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.carter.flat_env_cfg import CarterFlatEnvCfg

LOW_LEVEL_ENV_CFG = CarterFlatEnvCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=nav_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    # For Carter, we use direct velocity commands instead of a pre-trained policy
    # The low-level controller handles the wheel velocities
    velocity_command: nav_mdp.VelocityCommandActionCfg = nav_mdp.VelocityCommandActionCfg(
        asset_name="robot",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        wheel_base=0.413,  # Carter robot wheel base (m)
        wheel_radius=0.125,  # Carter robot wheel radius (m)
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=nav_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=nav_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=nav_mdp.projected_gravity)
        pose_command = ObsTerm(func=nav_mdp.generated_commands, params={"command_name": "pose_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Termination penalty
    termination_penalty = RewTerm(func=nav_mdp.is_terminated, weight=-400.0)

    # Position tracking rewards
    position_tracking = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 0.2, "command_name": "pose_command"},
    )

    # Orientation tracking
    orientation_tracking = RewTerm(
        func=nav_mdp.heading_command_error_abs,
        weight=-0.3,
        params={"command_name": "pose_command"},
    )

    # Penalize large linear velocity when close to goal
    velocity_near_goal = RewTerm(
        func=nav_mdp.velocity_near_goal_penalty,
        weight=-0.5,
        params={"distance_threshold": 0.5, "command_name": "pose_command"},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = nav_mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,  # Simplified heading for differential drive
        resampling_time_range=(10.0, 10.0),  # Give more time for navigation
        debug_vis=True,
        ranges=nav_mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-5.0, 5.0),
            pos_y=(-5.0, 5.0),
            heading=(-math.pi, math.pi)
        ),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=nav_mdp.time_out, time_out=True)

    # Terminate if robot tips over (base contacts ground)
    base_contact = DoneTerm(
        func=nav_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="chassis_link"), "threshold": 1.0},
    )


@configclass
class CarterNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Carter navigation environment."""

    # Scene settings from locomotion config
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene

    # MDP settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # Simulation settings
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation

        # Decimation: low-level runs at higher frequency, high-level at lower
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10

        # Episode length based on command resampling time
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        # Update sensor periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class CarterNavigationEnvCfg_PLAY(CarterNavigationEnvCfg):
    """Configuration for Carter navigation environment for evaluation."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False
