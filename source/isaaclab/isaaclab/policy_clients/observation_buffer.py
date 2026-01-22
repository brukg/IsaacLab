# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation buffer for temporal stacking of Gr00t observations."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch


class ObservationBuffer:
    """Buffer for managing temporal observation history for Gr00t inference.

    This buffer maintains separate histories for video and state observations,
    supporting different temporal horizons for each modality. It handles
    per-environment buffers for parallel simulation.

    Example:
        >>> buffer = ObservationBuffer(
        ...     num_envs=4,
        ...     video_horizon=2,
        ...     state_horizon=1,
        ...     video_keys=["wrist_camera", "scene_camera"],
        ...     state_keys=["joint_positions"],
        ...     device="cuda:0",
        ... )
        >>> # Append observations each step
        >>> buffer.append(video_obs, state_obs)
        >>> # Get stacked observations for Gr00t
        >>> video, state = buffer.get_stacked()
    """

    def __init__(
        self,
        num_envs: int,
        video_horizon: int = 1,
        state_horizon: int = 1,
        video_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        device: str = "cuda:0",
    ):
        """Initialize the observation buffer.

        Args:
            num_envs: Number of parallel environments.
            video_horizon: Number of frames to stack for video observations.
            state_horizon: Number of steps to stack for state observations.
            video_keys: List of camera names to track.
            state_keys: List of state names to track.
            device: Device for storing observations.
        """
        self.num_envs = num_envs
        self.video_horizon = video_horizon
        self.state_horizon = state_horizon
        self.video_keys = video_keys or []
        self.state_keys = state_keys or []
        self.device = device

        # Initialize buffers for each modality
        # Each buffer is a deque of tensors, one deque per key
        self._video_buffers: dict[str, deque[torch.Tensor]] = {
            key: deque(maxlen=video_horizon) for key in self.video_keys
        }
        self._state_buffers: dict[str, deque[torch.Tensor]] = {
            key: deque(maxlen=state_horizon) for key in self.state_keys
        }

        # Track which environments have been reset
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def append(
        self,
        video_obs: dict[str, torch.Tensor] | None = None,
        state_obs: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Append new observations to the buffer.

        Args:
            video_obs: Dictionary of camera observations.
                Each tensor has shape (num_envs, H, W, C).
            state_obs: Dictionary of state observations.
                Each tensor has shape (num_envs, D).
        """
        # Append video observations
        if video_obs is not None:
            for key in self.video_keys:
                if key in video_obs:
                    obs = video_obs[key].clone()
                    if not self._initialized.all():
                        # Initialize buffer with copies of first observation
                        while len(self._video_buffers[key]) < self.video_horizon:
                            self._video_buffers[key].append(obs.clone())
                    else:
                        self._video_buffers[key].append(obs)

        # Append state observations
        if state_obs is not None:
            for key in self.state_keys:
                if key in state_obs:
                    obs = state_obs[key].clone()
                    if not self._initialized.all():
                        # Initialize buffer with copies of first observation
                        while len(self._state_buffers[key]) < self.state_horizon:
                            self._state_buffers[key].append(obs.clone())
                    else:
                        self._state_buffers[key].append(obs)

        # Mark all environments as initialized
        self._initialized.fill_(True)

    def get_stacked(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Get temporally stacked observations in Gr00t format.

        Returns:
            Tuple of (video_dict, state_dict) where:
                - video_dict: {camera_name: np.ndarray(B, T, H, W, C), dtype=uint8}
                - state_dict: {state_name: np.ndarray(B, T, D), dtype=float32}

        Raises:
            RuntimeError: If buffer is empty (no observations appended yet).
        """
        if not self._initialized.any():
            raise RuntimeError("Buffer is empty. Call append() first.")

        video_stacked = {}
        state_stacked = {}

        # Stack video observations
        for key in self.video_keys:
            if len(self._video_buffers[key]) > 0:
                # Stack along temporal dimension: list of (B, H, W, C) -> (B, T, H, W, C)
                frames = list(self._video_buffers[key])
                stacked = torch.stack(frames, dim=1)  # (B, T, H, W, C)

                # Convert to numpy uint8
                if stacked.is_cuda:
                    stacked = stacked.cpu()

                if stacked.dtype == torch.float32:
                    video_stacked[key] = (stacked.numpy() * 255).astype(np.uint8)
                else:
                    video_stacked[key] = stacked.numpy().astype(np.uint8)

        # Stack state observations
        for key in self.state_keys:
            if len(self._state_buffers[key]) > 0:
                # Stack along temporal dimension: list of (B, D) -> (B, T, D)
                states = list(self._state_buffers[key])
                stacked = torch.stack(states, dim=1)  # (B, T, D)

                # Convert to numpy float32
                if stacked.is_cuda:
                    stacked = stacked.cpu()

                state_stacked[key] = stacked.numpy().astype(np.float32)

        return video_stacked, state_stacked

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        """Reset the buffer for specified environments.

        This clears the observation history for the specified environments,
        so they will be re-initialized on the next append.

        Args:
            env_ids: Environment indices to reset.
                If None, resets all environments.
        """
        if env_ids is None:
            # Reset all
            self._initialized.fill_(False)
            for key in self.video_keys:
                self._video_buffers[key].clear()
            for key in self.state_keys:
                self._state_buffers[key].clear()
        else:
            # For partial resets, we mark environments as needing re-initialization
            # The next append will handle filling the buffer appropriately
            if isinstance(env_ids, list):
                env_ids = torch.tensor(env_ids, device=self.device)
            self._initialized[env_ids] = False

            # Note: For partial resets with deque, we keep the buffer but
            # the observations for reset envs will be overwritten on next append
            # This is a simplification - for true per-env buffers we'd need
            # a more complex structure

    def reset_env(self, env_id: int) -> None:
        """Reset buffer for a single environment.

        Args:
            env_id: Index of the environment to reset.
        """
        self.reset(torch.tensor([env_id], device=self.device))

    @property
    def is_initialized(self) -> bool:
        """Check if the buffer has any observations."""
        return self._initialized.any().item()

    def get_video_buffer(self, key: str) -> list[torch.Tensor]:
        """Get the raw video buffer for a camera.

        Args:
            key: Camera name.

        Returns:
            List of observation tensors in temporal order (oldest first).
        """
        return list(self._video_buffers.get(key, []))

    def get_state_buffer(self, key: str) -> list[torch.Tensor]:
        """Get the raw state buffer for a state key.

        Args:
            key: State name.

        Returns:
            List of observation tensors in temporal order (oldest first).
        """
        return list(self._state_buffers.get(key, []))


class PerEnvObservationBuffer:
    """Per-environment observation buffer with proper isolation between environments.

    Unlike ObservationBuffer which uses shared deques, this class maintains
    completely separate buffers for each environment, allowing proper handling
    of environment resets.

    This is the recommended buffer for production use where environments
    may reset at different times.
    """

    def __init__(
        self,
        num_envs: int,
        video_horizon: int = 1,
        state_horizon: int = 1,
        video_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        device: str = "cuda:0",
    ):
        """Initialize the per-environment observation buffer.

        Args:
            num_envs: Number of parallel environments.
            video_horizon: Number of frames to stack for video observations.
            state_horizon: Number of steps to stack for state observations.
            video_keys: List of camera names to track.
            state_keys: List of state names to track.
            device: Device for storing observations.
        """
        self.num_envs = num_envs
        self.video_horizon = video_horizon
        self.state_horizon = state_horizon
        self.video_keys = video_keys or []
        self.state_keys = state_keys or []
        self.device = device

        # Per-env buffers: [env_id][key] = deque of tensors
        self._video_buffers: list[dict[str, deque[torch.Tensor]]] = [
            {key: deque(maxlen=video_horizon) for key in self.video_keys} for _ in range(num_envs)
        ]
        self._state_buffers: list[dict[str, deque[torch.Tensor]]] = [
            {key: deque(maxlen=state_horizon) for key in self.state_keys} for _ in range(num_envs)
        ]

        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def append(
        self,
        video_obs: dict[str, torch.Tensor] | None = None,
        state_obs: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Append new observations to the buffer for all environments.

        Args:
            video_obs: Dictionary of camera observations.
                Each tensor has shape (num_envs, H, W, C).
            state_obs: Dictionary of state observations.
                Each tensor has shape (num_envs, D).
        """
        for env_id in range(self.num_envs):
            is_first = not self._initialized[env_id].item()

            # Append video observations
            if video_obs is not None:
                for key in self.video_keys:
                    if key in video_obs:
                        obs = video_obs[key][env_id].clone()  # Single env obs
                        if is_first:
                            # Fill buffer with copies
                            while len(self._video_buffers[env_id][key]) < self.video_horizon:
                                self._video_buffers[env_id][key].append(obs.clone())
                        else:
                            self._video_buffers[env_id][key].append(obs)

            # Append state observations
            if state_obs is not None:
                for key in self.state_keys:
                    if key in state_obs:
                        obs = state_obs[key][env_id].clone()  # Single env obs
                        if is_first:
                            # Fill buffer with copies
                            while len(self._state_buffers[env_id][key]) < self.state_horizon:
                                self._state_buffers[env_id][key].append(obs.clone())
                        else:
                            self._state_buffers[env_id][key].append(obs)

            self._initialized[env_id] = True

    def get_stacked(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Get temporally stacked observations in Gr00t format.

        Returns:
            Tuple of (video_dict, state_dict) where:
                - video_dict: {camera_name: np.ndarray(B, T, H, W, C), dtype=uint8}
                - state_dict: {state_name: np.ndarray(B, T, D), dtype=float32}
        """
        video_stacked = {}
        state_stacked = {}

        # Stack video observations across all envs
        for key in self.video_keys:
            env_frames = []
            for env_id in range(self.num_envs):
                if len(self._video_buffers[env_id][key]) > 0:
                    # Stack temporal: list of (H, W, C) -> (T, H, W, C)
                    frames = list(self._video_buffers[env_id][key])
                    env_frames.append(torch.stack(frames, dim=0))

            if env_frames:
                # Stack across envs: list of (T, H, W, C) -> (B, T, H, W, C)
                stacked = torch.stack(env_frames, dim=0)
                if stacked.is_cuda:
                    stacked = stacked.cpu()
                if stacked.dtype == torch.float32:
                    video_stacked[key] = (stacked.numpy() * 255).astype(np.uint8)
                else:
                    video_stacked[key] = stacked.numpy().astype(np.uint8)

        # Stack state observations across all envs
        for key in self.state_keys:
            env_states = []
            for env_id in range(self.num_envs):
                if len(self._state_buffers[env_id][key]) > 0:
                    # Stack temporal: list of (D,) -> (T, D)
                    states = list(self._state_buffers[env_id][key])
                    env_states.append(torch.stack(states, dim=0))

            if env_states:
                # Stack across envs: list of (T, D) -> (B, T, D)
                stacked = torch.stack(env_states, dim=0)
                if stacked.is_cuda:
                    stacked = stacked.cpu()
                state_stacked[key] = stacked.numpy().astype(np.float32)

        return video_stacked, state_stacked

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        """Reset the buffer for specified environments.

        Args:
            env_ids: Environment indices to reset.
                If None, resets all environments.
        """
        if env_ids is None:
            env_ids = range(self.num_envs)
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.cpu().tolist()

        for env_id in env_ids:
            self._initialized[env_id] = False
            for key in self.video_keys:
                self._video_buffers[env_id][key].clear()
            for key in self.state_keys:
                self._state_buffers[env_id][key].clear()

    @property
    def is_initialized(self) -> bool:
        """Check if all environments have observations."""
        return self._initialized.all().item()
