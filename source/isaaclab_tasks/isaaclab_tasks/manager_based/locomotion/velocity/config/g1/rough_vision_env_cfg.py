# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Rough Terrain Environment Configuration with Depth Camera Support."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg, G1RoughEnvCfg_PLAY

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip


def process_depth_image(env, sensor_cfg: SceneEntityCfg, data_type: str = "distance_to_image_plane"):
    """Process depth camera observations.

    Processes raw depth camera data by clipping to a reasonable range,
    handling NaN/Inf values, and flattening for network input.
    Matches legged-loco implementation.
    """
    import torch

    sensor = env.scene.sensors[sensor_cfg.name].data
    output = sensor.output[data_type].clone()

    # Handle invalid values
    near_clip = 0.3
    far_clip = 5.0  # Extended range for better obstacle detection (was 2.0m)
    output[torch.isnan(output)] = far_clip
    output[torch.isinf(output)] = far_clip

    # Clip and normalize
    output = torch.clip(output, near_clip, far_clip)
    output = output - near_clip

    # Flatten for network input
    result = output.reshape(env.num_envs, -1)

    return result


@configclass
class G1VisionObservationsCfg:
    """Observation specifications for G1 with depth camera."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group (all observations combined for critic/standard policies)."""

        # Proprioceptive observations
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

        # Depth camera observation (for standard/critic observations)
        depth_measurement = ObsTerm(
            func=process_depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

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
class G1RoughVisionEnvCfg(G1RoughEnvCfg):
    """G1 rough terrain environment with depth camera support."""

    # Override observations with vision-enabled configuration
    observations: G1VisionObservationsCfg = G1VisionObservationsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Add depth camera to scene
        if not hasattr(self.scene, 'depth_camera'):
            # Create a new scene config that includes the depth camera
            from isaaclab.scene import InteractiveSceneCfg

            # Add depth camera sensor (matching legged-loco configuration)
            self.scene.depth_camera = RayCasterCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/pelvis",
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

        # Strengthen hip joint penalty to match legged-loco (better posture for vision control)
        self.rewards.joint_deviation_hip.weight = -0.2


@configclass
class G1RoughVisionEnvCfg_PLAY(G1RoughVisionEnvCfg):
    """G1 rough terrain vision environment for playing (inference)."""

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
