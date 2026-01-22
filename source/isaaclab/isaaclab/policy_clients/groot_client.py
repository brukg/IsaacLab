# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level client for Isaac-Gr00t VLA policy inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .data_transforms import build_groot_observation, format_language_instruction, groot_action_to_isaaclab
from .observation_buffer import PerEnvObservationBuffer
from .zmq_client import ModalityConfig, PolicyClient


class Gr00tPolicyClient:
    """High-level client for Isaac-Gr00t VLA (Vision-Language-Action) policy inference.

    This class provides a convenient interface for running Gr00t policies from IsaacLab.
    It handles:
    - Connection to the Gr00t inference server via ZMQ
    - Temporal observation buffering
    - Format conversion between IsaacLab (torch) and Gr00t (numpy)
    - Action extraction from Gr00t's action chunks

    Example:
        >>> # Create client
        >>> client = Gr00tPolicyClient(
        ...     host="localhost",
        ...     port=5555,
        ...     num_envs=4,
        ...     video_keys=["wrist_camera", "scene_camera"],
        ...     state_keys=["joint_positions"],
        ...     action_keys=["joint_positions", "gripper"],
        ... )
        >>>
        >>> # Main loop
        >>> obs, _ = env.reset()
        >>> while running:
        ...     action = client.get_action(obs, "pick up the red cube")
        ...     obs, reward, done, truncated, info = env.step(action)
        ...     if done.any():
        ...         client.reset(done.nonzero().squeeze())
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        num_envs: int = 1,
        video_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        action_keys: list[str] | None = None,
        groot_video_keys: list[str] | None = None,
        groot_state_keys: list[str] | None = None,
        video_horizon: int = 1,
        state_horizon: int = 1,
        action_horizon_idx: int = 0,
        device: str = "cuda:0",
        timeout_ms: int = 15000,
        api_token: str | None = None,
        auto_fetch_config: bool = True,
    ):
        """Initialize the Gr00t policy client.

        Args:
            host: Gr00t server hostname or IP address.
            port: Gr00t server port number.
            num_envs: Number of parallel environments.
            video_keys: Camera observation keys in IsaacLab (e.g., ["robot_pov_cam"]).
            state_keys: State observation keys in IsaacLab (e.g., ["robot_joint_pos"]).
            action_keys: Action output keys in order for concatenation.
            groot_video_keys: Corresponding Gr00t video keys (e.g., ["ego_view"]).
                If None, uses video_keys as-is.
            groot_state_keys: Corresponding Gr00t state keys (e.g., ["joint_position"]).
                If None, uses state_keys as-is.
            video_horizon: Number of frames to stack for video observations.
            state_horizon: Number of steps to stack for state observations.
            action_horizon_idx: Which action in the output horizon to use (0 = first).
            device: Torch device for action tensors.
            timeout_ms: ZMQ request timeout in milliseconds.
            api_token: Optional API token for authentication.
            auto_fetch_config: If True, fetch modality config from server on init.
        """
        self.host = host
        self.port = port
        self.num_envs = num_envs
        self.device = device
        self.action_horizon_idx = action_horizon_idx

        # Store keys - IsaacLab side
        self.video_keys = video_keys or []
        self.state_keys = state_keys or []
        self.action_keys = action_keys

        # Gr00t key mapping (for remapping to model's expected keys)
        self.groot_video_keys = groot_video_keys or self.video_keys
        self.groot_state_keys = groot_state_keys or self.state_keys

        # Build key mappings
        self._video_key_map = dict(zip(self.video_keys, self.groot_video_keys))
        self._state_key_map = dict(zip(self.state_keys, self.groot_state_keys))

        # Initialize ZMQ client
        self._client = PolicyClient(
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            api_token=api_token,
        )

        # Fetch modality config from server if requested
        self._modality_config: dict[str, ModalityConfig] | None = None
        if auto_fetch_config:
            try:
                self._modality_config = self._client.get_modality_config()
                # Update horizons from server config
                if "video" in self._modality_config:
                    video_horizon = len(self._modality_config["video"].delta_indices)
                if "state" in self._modality_config:
                    state_horizon = len(self._modality_config["state"].delta_indices)
                print(f"[Gr00tPolicyClient] Fetched modality config: video_horizon={video_horizon}, state_horizon={state_horizon}")
            except Exception as e:
                print(f"[Gr00tPolicyClient] Warning: Could not fetch modality config: {e}")

        self.video_horizon = video_horizon
        self.state_horizon = state_horizon

        # Initialize observation buffer
        self._obs_buffer = PerEnvObservationBuffer(
            num_envs=num_envs,
            video_horizon=video_horizon,
            state_horizon=state_horizon,
            video_keys=self.video_keys,
            state_keys=self.state_keys,
            device=device,
        )

        # Track connection status
        self._connected = False

    def connect(self) -> bool:
        """Test connection to the Gr00t server.

        Returns:
            True if server is reachable, False otherwise.
        """
        self._connected = self._client.ping()
        return self._connected

    @property
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        return self._connected

    @property
    def modality_config(self) -> dict[str, ModalityConfig] | None:
        """Get the modality configuration from the server."""
        return self._modality_config

    def get_action(
        self,
        observations: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        language_instruction: str | list[str],
        options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Get action from the Gr00t policy.

        Args:
            observations: Observation dictionary from IsaacLab environment.
                Expected structure (from env.step() or env.reset()):
                {
                    "policy": {
                        "wrist_camera": torch.Tensor(B, H, W, C),
                        "scene_camera": torch.Tensor(B, H, W, C),
                        "joint_positions": torch.Tensor(B, D),
                        ...
                    }
                }
                OR flat structure:
                {
                    "wrist_camera": torch.Tensor(B, H, W, C),
                    "joint_positions": torch.Tensor(B, D),
                    ...
                }
            language_instruction: Task instruction string or list of strings (one per env).
            options: Optional dictionary of additional options for the policy.

        Returns:
            Action tensor of shape (num_envs, action_dim) ready for env.step().

        Raises:
            RuntimeError: If server communication fails.
        """
        # Handle nested observation structure from IsaacLab
        if "policy" in observations:
            obs_dict = observations["policy"]
        else:
            obs_dict = observations

        # Ensure obs_dict values are tensors
        obs_dict = {k: v if isinstance(v, torch.Tensor) else v for k, v in obs_dict.items()}

        # Extract video and state observations
        video_obs = {k: obs_dict[k] for k in self.video_keys if k in obs_dict}
        state_obs = {k: obs_dict[k] for k in self.state_keys if k in obs_dict}

        # Append to temporal buffer
        self._obs_buffer.append(video_obs, state_obs)

        # Get stacked observations
        video_stacked, state_stacked = self._obs_buffer.get_stacked()

        # Remap keys from IsaacLab names to Gr00t expected names
        video_remapped = {self._video_key_map[k]: v for k, v in video_stacked.items()}
        state_remapped = {self._state_key_map[k]: v for k, v in state_stacked.items()}

        # Format language instruction
        language = format_language_instruction(
            language_instruction,
            num_envs=self.num_envs,
            temporal_horizon=1,
        )

        # Build complete Gr00t observation
        groot_obs = build_groot_observation(video_remapped, state_remapped, language)

        # Call Gr00t server
        action_dict, info = self._client.get_action(groot_obs, options)

        # Convert action to IsaacLab format
        action_tensor = groot_action_to_isaaclab(
            action_dict,
            action_keys=self.action_keys,
            action_horizon_idx=self.action_horizon_idx,
            device=self.device,
        )

        return action_tensor

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        """Reset the observation buffer for specified environments.

        Call this when environments are reset to clear their observation history.

        Args:
            env_ids: Environment indices to reset.
                If None, resets all environments.
        """
        self._obs_buffer.reset(env_ids)

        # Optionally notify the server about reset
        if env_ids is None:
            self._client.reset()

    def reset_policy(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy state on the server.

        This resets any internal state the Gr00t policy may have
        (e.g., RNN hidden states).

        Args:
            options: Optional reset options.

        Returns:
            Reset information from the server.
        """
        return self._client.reset(options)

    def close(self) -> None:
        """Close the connection to the Gr00t server."""
        self._client.close()
        self._connected = False

    def __enter__(self) -> "Gr00tPolicyClient":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Cleanup on destruction."""
        try:
            self.close()
        except Exception:
            pass


def create_groot_client_from_env(
    env: Any,
    host: str = "localhost",
    port: int = 5555,
    video_keys: list[str] | None = None,
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
    **kwargs,
) -> Gr00tPolicyClient:
    """Create a Gr00tPolicyClient from an IsaacLab environment.

    This is a convenience function that extracts environment properties
    to configure the client.

    Args:
        env: IsaacLab environment instance.
        host: Gr00t server hostname.
        port: Gr00t server port.
        video_keys: Camera observation keys (auto-detected if None).
        state_keys: State observation keys (auto-detected if None).
        action_keys: Action keys (auto-detected if None).
        **kwargs: Additional arguments passed to Gr00tPolicyClient.

    Returns:
        Configured Gr00tPolicyClient instance.
    """
    # Get number of environments
    num_envs = env.num_envs

    # Get device
    device = str(env.device)

    # Auto-detect video keys from environment config if available
    if video_keys is None and hasattr(env, "cfg"):
        cfg = env.cfg
        if hasattr(cfg, "scene"):
            # Look for camera sensors in scene
            video_keys = []
            for name, entity in vars(cfg.scene).items():
                if hasattr(entity, "data_types") and "rgb" in getattr(entity, "data_types", []):
                    video_keys.append(name)

    # Create and return client
    return Gr00tPolicyClient(
        host=host,
        port=port,
        num_envs=num_envs,
        video_keys=video_keys or [],
        state_keys=state_keys or [],
        action_keys=action_keys,
        device=device,
        **kwargs,
    )
