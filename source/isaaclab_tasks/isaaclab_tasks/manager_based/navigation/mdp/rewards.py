# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


def velocity_near_goal_penalty(
    env: ManagerBasedRLEnv, distance_threshold: float, command_name: str
) -> torch.Tensor:
    """Penalize high velocities when close to the goal position.

    This encourages the robot to slow down as it approaches the target.
    """
    from isaaclab.assets import Articulation

    # Get the robot asset
    robot: Articulation = env.scene["robot"]

    # Get command to compute distance to goal
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b[:, :2], dim=1)  # Only XY distance

    # Get current velocity magnitude
    velocity = robot.data.root_lin_vel_b
    velocity_magnitude = torch.norm(velocity[:, :2], dim=1)  # Only XY velocity

    # Apply penalty only when close to goal
    near_goal = distance < distance_threshold
    penalty = torch.where(near_goal, velocity_magnitude, torch.zeros_like(velocity_magnitude))

    return penalty
