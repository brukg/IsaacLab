# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 Navigation Environment Configuration with Vision Support."""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
import isaaclab_tasks.manager_based.navigation.mdp as navigation_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg

# Use H1 flat terrain as the low-level environment for flat navigation
LOW_LEVEL_FLAT_ENV_CFG = H1FlatEnvCfg()
# Use H1 rough terrain as the low-level environment for rough navigation
LOW_LEVEL_ROUGH_ENV_CFG = H1RoughEnvCfg()


def process_depth_image(env, sensor_cfg: SceneEntityCfg, data_type: str = "distance_to_image_plane"):
    """Process depth camera observations as 2D images.

    Processes raw depth camera data by clipping to a reasonable range,
    handling NaN/Inf values, and returning as 2D images for CNN processing.

    Returns:
        torch.Tensor: Depth images of shape (B, 1, H, W) where B is batch size,
                     1 is the channel dimension, H=53 and W=30 are image dimensions.
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
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=locomotion_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsFlatCfg:
    """Action terms for flat terrain navigation."""

    pre_trained_policy_action: navigation_mdp.PreTrainedPolicyActionCfg = navigation_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/H1/Flat/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_FLAT_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_FLAT_ENV_CFG.observations.policy,
    )


@configclass
class ActionsRoughCfg:
    """Action terms for rough terrain navigation."""

    pre_trained_policy_action: navigation_mdp.PreTrainedPolicyActionCfg = navigation_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/H1/Rough/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ROUGH_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ROUGH_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (base navigation, no vision)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Basic proprioception for navigation
        base_lin_vel = ObsTerm(func=locomotion_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=locomotion_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=locomotion_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # Navigation goal
        pose_command = ObsTerm(func=locomotion_mdp.generated_commands, params={"command_name": "pose_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=locomotion_mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=navigation_mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=navigation_mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=navigation_mdp.heading_command_error_abs,
        weight=-0.3,
        params={"command_name": "pose_command"},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = navigation_mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=navigation_mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-5.0, 5.0), pos_y=(-5.0, 5.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=locomotion_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=locomotion_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )


@configclass
class H1NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for H1 flat navigation environment (base, no vision)."""

    # environment settings
    scene: SceneEntityCfg = LOW_LEVEL_FLAT_ENV_CFG.scene
    actions: ActionsFlatCfg = ActionsFlatCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_FLAT_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_FLAT_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_FLAT_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        # Override policy path to use local trained policy
        # Users can override this by setting policy_path before calling __post_init__


        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class H1NavigationEnvCfg_PLAY(H1NavigationEnvCfg):
    """H1 navigation environment for playing (inference)."""

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# ============================================================================
# Vision Variants
# ============================================================================


@configclass
class H1NavigationDepthObservationsCfg:
    """Observation specifications with DEPTH CAMERA ONLY."""

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
class H1NavigationScannerObservationsCfg:
    """Observation specifications with HEIGHT SCANNER ONLY."""

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

        # Height scanner for terrain awareness
        height_scan = ObsTerm(
            func=locomotion_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

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
    class ScannerCfg(ObsGroup):
        """Height scanner observations only."""

        height_scan = ObsTerm(
            func=locomotion_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioceptionCfg = ProprioceptionCfg()
    scanner: ScannerCfg = ScannerCfg()


@configclass
class H1NavigationDepthScannerObservationsCfg:
    """Observation specifications with DEPTH CAMERA + HEIGHT SCANNER."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (includes height scanner for critic)."""

        # Basic proprioception
        base_lin_vel = ObsTerm(func=locomotion_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=locomotion_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=locomotion_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_command = ObsTerm(func=locomotion_mdp.generated_commands, params={"command_name": "pose_command"})

        # Height scanner (used by critic for better value estimation)
        height_scan = ObsTerm(
            func=locomotion_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

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

    @configclass
    class ScannerCfg(ObsGroup):
        """Height scanner observations only."""

        height_scan = ObsTerm(
            func=locomotion_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioceptionCfg = ProprioceptionCfg()
    depth: DepthCfg = DepthCfg()
    scanner: ScannerCfg = ScannerCfg()


# ============================================================================
# Environment Configurations - Depth Camera Only
# ============================================================================


@configclass
class H1NavigationDepthEnvCfg(H1NavigationEnvCfg):
    """H1 navigation environment with DEPTH CAMERA ONLY."""

    observations: H1NavigationDepthObservationsCfg = H1NavigationDepthObservationsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Add depth camera to scene
        if not hasattr(self.scene, 'depth_camera'):
            self.scene.depth_camera = RayCasterCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link",
                mesh_prim_paths=["/World/ground"],
                offset=RayCasterCameraCfg.OffsetCfg(pos=(0.108, -0.0325, 0.420), rot=(0.389, 0.0, 0.921, 0.0)),
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
class H1NavigationDepthEnvCfg_PLAY(H1NavigationDepthEnvCfg):
    """H1 navigation depth camera environment for playing (inference)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ============================================================================
# Environment Configurations - Height Scanner Only
# ============================================================================


@configclass
class H1NavigationScannerEnvCfg(H1NavigationEnvCfg):
    """H1 navigation environment with HEIGHT SCANNER ONLY."""

    observations: H1NavigationScannerObservationsCfg = H1NavigationScannerObservationsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Height scanner is already in LOW_LEVEL_ENV_CFG.scene
        # Just ensure it's configured properly
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )


@configclass
class H1NavigationScannerEnvCfg_PLAY(H1NavigationScannerEnvCfg):
    """H1 navigation height scanner environment for playing (inference)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ============================================================================
# Environment Configurations - Depth + Scanner Combined
# ============================================================================


@configclass
class H1NavigationDepthScannerEnvCfg(H1NavigationEnvCfg):
    """H1 navigation environment with DEPTH CAMERA + HEIGHT SCANNER."""

    observations: H1NavigationDepthScannerObservationsCfg = H1NavigationDepthScannerObservationsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Add depth camera to scene
        if not hasattr(self.scene, 'depth_camera'):
            self.scene.depth_camera = RayCasterCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link",
                mesh_prim_paths=["/World/ground"],
                offset=RayCasterCameraCfg.OffsetCfg(pos=(0.108, -0.0325, 0.420), rot=(0.389, 0.0, 0.921, 0.0)),
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

        # Ensure height scanner is configured
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )


@configclass
class H1NavigationDepthScannerEnvCfg_PLAY(H1NavigationDepthScannerEnvCfg):
    """H1 navigation depth+scanner environment for playing (inference)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ============================================================================
# ROUGH TERRAIN NAVIGATION ENVIRONMENTS
# ============================================================================


@configclass
class H1NavigationRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for H1 rough terrain navigation environment with height scanner."""

    # environment settings
    scene: SceneEntityCfg = LOW_LEVEL_ROUGH_ENV_CFG.scene
    actions: ActionsRoughCfg = ActionsRoughCfg()
    observations: H1NavigationScannerObservationsCfg = H1NavigationScannerObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ROUGH_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ROUGH_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ROUGH_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class H1NavigationRoughEnvCfg_PLAY(H1NavigationRoughEnvCfg):
    """H1 rough terrain navigation environment for playing (inference)."""

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# ============================================================================
# Rough Terrain - Vision (Depth Camera)
# ============================================================================


@configclass
class H1NavigationRoughVisionEnvCfg(H1NavigationRoughEnvCfg):
    """H1 rough terrain navigation environment with vision (depth camera)."""

    observations: H1NavigationDepthObservationsCfg = H1NavigationDepthObservationsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Override terrain with vision-optimized configuration (boxes and obstacles for depth camera)
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
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.15,
                    step_height_range=(0.05, 0.30),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.15,
                    step_height_range=(0.05, 0.30),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                # Box obstacles for depth camera
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.35,
                    grid_width=0.45,
                    grid_height_range=(0.05, 0.3),
                    platform_width=2.0,
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.05,
                    noise_range=(0.02, 0.10),
                    noise_step=0.02,
                    border_width=0.25,
                ),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.05,
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.05,
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                # Giant 1.5m tall obstacles for challenging navigation
                "init_pos": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                    proportion=0.2,
                    obstacle_height_mode="choice",
                    obstacle_height_range=(1.5, 1.5),
                    obstacle_width_range=(0.3, 1.5),
                    num_obstacles=900,
                    platform_width=1.5,
                ),
            },
        )

        # Add depth camera to scene
        if not hasattr(self.scene, 'depth_camera'):
            self.scene.depth_camera = RayCasterCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link",
                mesh_prim_paths=["/World/ground"],
                offset=RayCasterCameraCfg.OffsetCfg(pos=(0.108, -0.0325, 0.420), rot=(0.389, 0.0, 0.921, 0.0)),
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
class H1NavigationRoughVisionEnvCfg_PLAY(H1NavigationRoughVisionEnvCfg):
    """H1 rough terrain navigation vision environment for playing (inference)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ============================================================================
# Rough Terrain - Vision + Scanner Combined
# ============================================================================


@configclass
class H1NavigationRoughVisionScannerEnvCfg(H1NavigationRoughVisionEnvCfg):
    """H1 rough terrain navigation environment with vision (depth camera) + height scanner."""

    observations: H1NavigationDepthScannerObservationsCfg = H1NavigationDepthScannerObservationsCfg()

    def __post_init__(self):
        # Post init of parent (inherits terrain config and depth camera from Vision)
        super().__post_init__()

        # Ensure height scanner is configured
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )


@configclass
class H1NavigationRoughVisionScannerEnvCfg_PLAY(H1NavigationRoughVisionScannerEnvCfg):
    """H1 rough terrain navigation vision+scanner environment for playing (inference)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
