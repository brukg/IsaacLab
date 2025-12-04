# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity command action for differential drive robots in navigation tasks."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class VelocityCommandAction(ActionTerm):
    r"""Velocity command action term for differential drive robots.

    This action term converts high-level velocity commands (linear_x, linear_y, angular_z)
    into low-level wheel velocities for differential drive robots. It wraps the low-level
    locomotion controller to handle the conversion.

    For navigation tasks, the high-level policy outputs desired velocities, and this term
    uses the low-level locomotion policy to achieve those velocities.

    """

    cfg: VelocityCommandActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: VelocityCommandActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        def last_action():
            # reset the low level actions if the episode was reset
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.velocity_commands.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

        # For differential drive, we'll use a simple conversion from velocity commands to wheel velocities
        # This is a placeholder for a more sophisticated controller
        self.wheel_base = cfg.wheel_base  # distance between wheels
        self.wheel_radius = cfg.wheel_radius

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3  # linear_x, linear_y (unused for diff drive), angular_z

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            # Get low-level observations
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            # Convert velocity commands to wheel velocities using differential drive kinematics
            # For differential drive:
            # v_left = v_linear - (v_angular * wheel_base / 2)
            # v_right = v_linear + (v_angular * wheel_base / 2)
            # Then convert linear velocity to angular velocity: omega = v / r

            v_linear = self._raw_actions[:, 0]  # forward velocity
            v_angular = self._raw_actions[:, 2]  # yaw rate

            # Calculate wheel linear velocities
            v_left = v_linear - (v_angular * self.wheel_base / 2.0)
            v_right = v_linear + (v_angular * self.wheel_base / 2.0)

            # Convert to wheel angular velocities (rad/s)
            omega_left = v_left / self.wheel_radius
            omega_right = v_right / self.wheel_radius

            # Stack wheel velocities for low-level action
            self.low_level_actions[:, 0] = omega_left
            self.low_level_actions[:, 1] = omega_right

            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0

        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class VelocityCommandActionCfg(ActionTermCfg):
    """Configuration for velocity command action term.

    See :class:`VelocityCommandAction` for more details.
    """

    class_type: type[ActionTerm] = VelocityCommandAction
    """Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    wheel_base: float = 0.413
    """Distance between left and right wheels (m). Default is for Carter robot."""
    wheel_radius: float = 0.125
    """Radius of the wheels (m). Default is for Carter robot."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to True."""
