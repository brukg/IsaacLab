# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 Rough Terrain Environment Configuration with Depth Camera Support."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ImuCfg
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg, H1RoughEnvCfg_PLAY


def process_depth_image(env, sensor_cfg: SceneEntityCfg, data_type: str = "distance_to_image_plane"):
    """Process depth camera observations.

    Processes raw depth camera data by clipping to a reasonable range,
    handling NaN/Inf values, and flattening for network input.
    """
    import torch

    sensor = env.scene.sensors[sensor_cfg.name].data
    output = sensor.output[data_type].clone()

    # Handle invalid values
    near_clip = 0.3
    far_clip = 5.0  # Extended range for better obstacle detection
    output[torch.isnan(output)] = far_clip
    output[torch.isinf(output)] = far_clip

    # Clip and normalize
    output = torch.clip(output, near_clip, far_clip)
    output = output - near_clip

    # Flatten for network input
    result = output.reshape(env.num_envs, -1)

    return result


@configclass
class H1VisionObservationsCfg:
    """Observation specifications for H1 with depth camera."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (all observations combined for critic/standard policies)."""

        # Proprioceptive observations (ground truth)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        # Height scanner for terrain awareness (critic uses this)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioceptionCfg(ObsGroup):
        """Proprioceptive observations only (for history wrapper)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class DepthCfg(ObsGroup):
        """Depth camera observations only."""

        depth_measurement = ObsTerm(
            func=process_depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioceptionCfg = ProprioceptionCfg()
    depth: DepthCfg = DepthCfg()


@configclass
class H1RoughVisionEnvCfg(H1RoughEnvCfg):
    """H1 rough terrain environment with depth camera support.

    Matches legged-loco H1 vision configuration with challenging terrain including
    1.5m tall obstacles for vision-based locomotion.
    """

    # Override observations with vision-enabled configuration
    observations: H1VisionObservationsCfg = H1VisionObservationsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Override terrain with legged-loco H1 vision configuration
        # Heavy emphasis on discrete obstacles for depth camera usage
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
                    proportion=0.15,  # Reduced from 0.2
                    step_height_range=(0.05, 0.30),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.15,  # Reduced from 0.2
                    step_height_range=(0.05, 0.30),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                # INCREASED: More box obstacles everywhere for depth camera
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.35,  # INCREASED from 0.2 to 0.35
                    grid_width=0.45,
                    grid_height_range=(0.05, 0.3),  # Slightly taller boxes
                    platform_width=2.0,
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.05,  # Reduced from 0.2 to make room for boxes
                    noise_range=(0.02, 0.10),
                    noise_step=0.02,
                    border_width=0.25,
                ),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.05,  # Reduced from 0.1
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.05,  # Reduced from 0.1
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                # Giant 1.5m tall obstacles - keep at 20% for challenging areas
                "init_pos": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                    proportion=0.4,
                    obstacle_height_mode="choice",
                    obstacle_height_range=(1.5, 1.5),  # 1.5m tall obstacles!
                    obstacle_width_range=(0.3, 1.5),
                    num_obstacles=900,  # Increased from 30 for more density
                    platform_width=1.5,
                ),
            },
        )

        # Add depth camera to scene
        if not hasattr(self.scene, 'depth_camera'):
            self.scene.depth_camera = RayCasterCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link",
                mesh_prim_paths=["/World/ground"],
                offset=RayCasterCameraCfg.OffsetCfg(
                    pos=(0.108, -0.0325, 0.420),
                    rot=(0.389, 0.0, 0.921, 0.0)
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

        # Add IMU sensor to torso
        if not hasattr(self.scene, 'imu'):
            self.scene.imu = ImuCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link",
                update_period=0.0,  # Update every physics step
                debug_vis=False,
            )


@configclass
class H1RoughVisionEnvCfg_PLAY(H1RoughVisionEnvCfg):
    """H1 rough terrain vision environment for playing (inference)."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # Spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # Reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # Disable randomization for play
        self.observations.policy.enable_corruption = False
        # Remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


##
# IMU Sensor Variant (for realistic state estimation)
##


@configclass
class H1VisionIMUObservationsCfg(H1VisionObservationsCfg):
    """Observation specifications for H1 with depth camera and IMU sensor."""

    @configclass
    class PolicyCfg(H1VisionObservationsCfg.PolicyCfg):
        """Policy observations using IMU instead of ground truth."""

        # Override ground truth with IMU observations
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        imu_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.02, n_max=0.02)
        )

        # Remove ground truth observations
        base_ang_vel: ObsTerm = None
        projected_gravity: ObsTerm = None

    @configclass
    class ProprioceptionCfg(H1VisionObservationsCfg.ProprioceptionCfg):
        """Proprioceptive observations using IMU."""

        # Override with IMU
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        imu_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.02, n_max=0.02)
        )

        # Remove ground truth
        base_ang_vel: ObsTerm = None
        projected_gravity: ObsTerm = None


@configclass
class H1RoughVisionIMUEnvCfg(H1RoughVisionEnvCfg):
    """H1 rough terrain with depth camera and IMU sensor (realistic state estimation)."""

    # Use IMU observations
    observations: H1VisionIMUObservationsCfg = H1VisionIMUObservationsCfg()


@configclass
class H1RoughVisionIMUEnvCfg_PLAY(H1RoughVisionIMUEnvCfg):
    """H1 rough terrain vision + IMU environment for playing (inference)."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # Spawn the robot randomly in the grid
        self.scene.terrain.max_init_terrain_level = None
        # Reduce the number of terrains
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # Disable randomization for play
        self.observations.policy.enable_corruption = False
        # Remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
