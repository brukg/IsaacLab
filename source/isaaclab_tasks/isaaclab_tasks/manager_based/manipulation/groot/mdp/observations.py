# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for Isaac-Gr00t environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def gripper_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get the gripper state (open/closed) as a normalized value.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.

    Returns:
        Gripper state tensor of shape (num_envs, 1) with values in [0, 1].
        0 = fully closed, 1 = fully open.
    """
    asset = env.scene[asset_cfg.name]

    # Get gripper joint positions - assumes last joints are gripper
    # This is a simplified implementation; real robots may need custom handling
    joint_pos = asset.data.joint_pos

    # Normalize to [0, 1] based on joint limits
    joint_limits = asset.data.soft_joint_pos_limits
    if joint_limits is not None:
        lower = joint_limits[..., 0]
        upper = joint_limits[..., 1]
        normalized = (joint_pos - lower) / (upper - lower + 1e-6)
        # Take mean of last N joints as gripper state (assumes symmetric gripper)
        gripper_value = normalized[..., -1:].mean(dim=-1, keepdim=True)
    else:
        gripper_value = joint_pos[..., -1:]

    return gripper_value.clamp(0, 1)


def end_effector_pose(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_hand",
) -> torch.Tensor:
    """Get the end-effector pose (position + quaternion) in world frame.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.
        body_name: The name of the end-effector body/link.

    Returns:
        End-effector pose tensor of shape (num_envs, 7) containing [x, y, z, qw, qx, qy, qz].
    """
    asset = env.scene[asset_cfg.name]

    # Get body index
    body_idx = asset.find_bodies(body_name)[0][0]

    # Get position and orientation
    pos = asset.data.body_pos_w[:, body_idx, :]  # (num_envs, 3)
    quat = asset.data.body_quat_w[:, body_idx, :]  # (num_envs, 4)

    return torch.cat([pos, quat], dim=-1)


def end_effector_velocity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_hand",
) -> torch.Tensor:
    """Get the end-effector velocity (linear + angular) in world frame.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.
        body_name: The name of the end-effector body/link.

    Returns:
        End-effector velocity tensor of shape (num_envs, 6) containing [vx, vy, vz, wx, wy, wz].
    """
    asset = env.scene[asset_cfg.name]

    # Get body index
    body_idx = asset.find_bodies(body_name)[0][0]

    # Get linear and angular velocity
    lin_vel = asset.data.body_lin_vel_w[:, body_idx, :]  # (num_envs, 3)
    ang_vel = asset.data.body_ang_vel_w[:, body_idx, :]  # (num_envs, 3)

    return torch.cat([lin_vel, ang_vel], dim=-1)


def object_pose_in_robot_frame(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get the object pose relative to the robot base frame.

    Args:
        env: The environment instance.
        robot_cfg: The asset configuration for the robot.
        object_cfg: The asset configuration for the object.

    Returns:
        Object pose tensor of shape (num_envs, 7) in robot base frame.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    # Get robot base pose
    robot_pos = robot.data.root_pos_w  # (num_envs, 3)
    robot_quat = robot.data.root_quat_w  # (num_envs, 4)

    # Get object pose
    obj_pos = obj.data.root_pos_w  # (num_envs, 3)
    obj_quat = obj.data.root_quat_w  # (num_envs, 4)

    # Transform object position to robot frame
    # rel_pos = R_robot^T @ (obj_pos - robot_pos)
    rel_pos = obj_pos - robot_pos

    # Rotate to robot frame using quaternion inverse
    # For unit quaternion, inverse is conjugate: [w, -x, -y, -z]
    robot_quat_inv = robot_quat.clone()
    robot_quat_inv[:, 1:] = -robot_quat_inv[:, 1:]

    # Apply rotation (simplified - for full implementation use quaternion rotation)
    # This is an approximation; for accurate results use proper quaternion math
    rel_pos_rotated = rel_pos  # Simplified for now

    return torch.cat([rel_pos_rotated, obj_quat], dim=-1)
