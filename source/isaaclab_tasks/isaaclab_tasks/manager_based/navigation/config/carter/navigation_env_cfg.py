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
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
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
    """Action terms for the MDP using pre-trained locomotion policy."""

    # Use pre-trained locomotion policy for low-level control
    pre_trained_policy_action: nav_mdp.PreTrainedPolicyActionCfg = nav_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path="POLICY_PATH_PLACEHOLDER",  # Must be set via --load_run argument
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
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
    """Configuration for the Carter navigation environment.

    This environment requires a pre-trained locomotion policy. Train it first:
    1. Train locomotion: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Carter-v0
    2. Train navigation: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Navigation-Flat-Carter-v0 \\
                         --low_level_policy logs/rsl_rl/carter_flat/exported/policy.pt
    """

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


# ============================================================================
# Vision-Based Navigation with Depth Camera
# ============================================================================


def process_depth_image(env, sensor_cfg: SceneEntityCfg, data_type: str = "distance_to_image_plane"):
    """Process depth camera observations as 2D images.

    Processes raw depth camera data by clipping to a reasonable range,
    handling NaN/Inf values, and returning as 2D images for CNN processing.

    Returns:
        torch.Tensor: Depth images of shape (B, 1, H, W) where B is batch size,
                     1 is the channel dimension, H and W are image dimensions.
    """
    import torch

    sensor = env.scene.sensors[sensor_cfg.name].data
    output = sensor.output[data_type].clone()

    # Handle invalid values
    near_clip = 0.3
    far_clip = 5.0
    output[torch.isnan(output)] = far_clip
    output[torch.isinf(output)] = far_clip

    # Clip and normalize
    output = torch.clip(output, near_clip, far_clip)
    output = output - near_clip

    # Check if output already has a channel dimension at the end
    # RayCasterCamera returns (B, H, W, 1) for single-channel data
    if output.ndim == 4 and output.shape[-1] == 1:
        # Remove the trailing channel dimension and add it at position 1
        # (B, H, W, 1) -> (B, H, W) -> (B, 1, H, W)
        output = output.squeeze(-1).unsqueeze(1)
    elif output.ndim == 3:
        # (B, H, W) -> (B, 1, H, W)
        output = output.unsqueeze(1)
    else:
        raise RuntimeError(f"Unexpected depth sensor output shape: {output.shape}")

    return output


@configclass
class CarterNavigationDepthObservationsCfg:
    """Observation specifications with DEPTH CAMERA for Carter navigation."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Basic proprioception
        base_lin_vel = ObsTerm(func=locomotion_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=locomotion_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=locomotion_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_command = ObsTerm(func=locomotion_mdp.generated_commands, params={"command_name": "pose_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioceptionCfg(ObsGroup):
        """Proprioceptive observations only."""

        base_lin_vel = ObsTerm(func=locomotion_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=locomotion_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=locomotion_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_command = ObsTerm(func=locomotion_mdp.generated_commands, params={"command_name": "pose_command"})

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class DepthCfg(ObsGroup):
        """Depth camera observations only."""

        depth_measurement = ObsTerm(
            func=process_depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane"},
        )

        def __post_init__(self):
            self.concatenate_terms = False  # Return dict, extracted by runner

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioceptionCfg = ProprioceptionCfg()
    depth: DepthCfg = DepthCfg()


@configclass
class CarterNavigationDepthActionsCfg:
    """Action terms for vision-based navigation using pre-trained locomotion policy."""

    pre_trained_policy_action: nav_mdp.PreTrainedPolicyActionCfg = nav_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path="POLICY_PATH_PLACEHOLDER",  # Must be set via --load_run argument
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class CarterNavigationDepthEnvCfg(ManagerBasedRLEnvCfg):
    """Carter navigation environment with depth camera and obstacles on flat terrain.

    This environment requires a pre-trained locomotion policy. Train it first:
    1. Train locomotion: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Carter-v0
    2. Train navigation: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Navigation-Flat-Carter-Depth-v0 \\
                         --low_level_policy logs/rsl_rl/carter_flat/exported/policy.pt
    """

    # Scene settings from locomotion config
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene

    # Use pre-trained policy actions instead of direct velocity commands
    actions: CarterNavigationDepthActionsCfg = CarterNavigationDepthActionsCfg()
    observations: CarterNavigationDepthObservationsCfg = CarterNavigationDepthObservationsCfg()
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

        # Override terrain to add box obstacles for vision-based navigation
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                # Random grid of boxes - primary obstacle type
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.6,
                    grid_width=0.45,
                    grid_height_range=(0.1, 0.5),  # Boxes between 10-50cm tall
                    platform_width=2.0,
                ),
                # Discrete obstacles - scattered boxes
                "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                    proportion=0.3,
                    obstacle_height_mode="choice",
                    obstacle_height_range=(0.2, 0.6),  # 20-60cm tall obstacles
                    obstacle_width_range=(0.3, 0.8),
                    num_obstacles=50,
                    platform_width=2.0,
                ),
                # Some flat areas for easier navigation
                "flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.1,
                    noise_range=(0.0, 0.01),  # Nearly flat
                    noise_step=0.01,
                    border_width=0.25,
                ),
            },
        )

        # Add depth camera mounted on Carter chassis
        # Carter chassis_link is the main body
        if not hasattr(self.scene, 'depth_camera'):
            self.scene.depth_camera = RayCasterCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/chassis_link",
                mesh_prim_paths=["/World/ground"],
                # Offset from chassis center - forward facing camera
                # Carter chassis is approximately 0.4m long, mount camera at front
                offset=RayCasterCameraCfg.OffsetCfg(
                    pos=(0.25, 0.0, 0.15),  # Forward 25cm, up 15cm from chassis center
                    rot=(1.0, 0.0, 0.0, 0.0),  # No rotation, pointing forward
                ),
                data_types=["distance_to_image_plane"],
                debug_vis=False,
                pattern_cfg=patterns.PinholeCameraPatternCfg(
                    focal_length=1.93,
                    horizontal_aperture=3.8,
                    height=53,
                    width=30,
                ),
                max_distance=10.0,
            )


@configclass
class CarterNavigationDepthEnvCfg_PLAY(CarterNavigationDepthEnvCfg):
    """Carter navigation depth camera environment for evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
