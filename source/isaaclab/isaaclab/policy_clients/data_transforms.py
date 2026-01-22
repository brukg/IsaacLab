# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data transformation utilities for converting between IsaacLab and Gr00t formats."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def isaaclab_images_to_groot(
    images: dict[str, torch.Tensor],
    temporal_stack: dict[str, list[torch.Tensor]] | None = None,
) -> dict[str, np.ndarray]:
    """Convert IsaacLab camera images to Gr00t video format.

    Args:
        images: Dictionary mapping camera names to image tensors.
            Expected shape: (B, H, W, C) with dtype uint8 or float32.
            If float32, values should be in [0, 1] range.
        temporal_stack: Optional dictionary of previous images for temporal stacking.
            If provided, images will be stacked along temporal dimension.

    Returns:
        Dictionary mapping camera names to numpy arrays of shape (B, T, H, W, 3)
        with dtype uint8 and values in [0, 255] range.

    Example:
        >>> images = {"wrist_camera": torch.zeros(4, 224, 224, 3, dtype=torch.uint8)}
        >>> groot_video = isaaclab_images_to_groot(images)
        >>> groot_video["wrist_camera"].shape
        (4, 1, 224, 224, 3)
    """
    result = {}

    for camera_name, image_tensor in images.items():
        # Ensure tensor is on CPU and convert to numpy
        if image_tensor.is_cuda:
            image_tensor = image_tensor.cpu()

        # Convert to uint8 if needed
        if image_tensor.dtype == torch.float32:
            # Assume float values are in [0, 1] range
            image_np = (image_tensor.numpy() * 255).astype(np.uint8)
        else:
            image_np = image_tensor.numpy().astype(np.uint8)

        # Ensure shape is (B, H, W, C)
        if image_np.ndim == 3:
            # Single image (H, W, C) -> (1, H, W, C)
            image_np = image_np[np.newaxis, ...]

        # Stack temporally if previous images are provided
        if temporal_stack is not None and camera_name in temporal_stack:
            # Stack previous images with current
            prev_images = temporal_stack[camera_name]
            # Each prev image should be (B, H, W, C)
            all_images = [img.cpu().numpy() if isinstance(img, torch.Tensor) else img for img in prev_images]
            all_images.append(image_np)
            # Stack along new temporal dimension: (B, T, H, W, C)
            image_np = np.stack(all_images, axis=1)
        else:
            # Add temporal dimension: (B, H, W, C) -> (B, 1, H, W, C)
            image_np = image_np[:, np.newaxis, ...]

        result[camera_name] = image_np

    return result


def isaaclab_state_to_groot(
    states: dict[str, torch.Tensor],
    temporal_stack: dict[str, list[torch.Tensor]] | None = None,
) -> dict[str, np.ndarray]:
    """Convert IsaacLab state tensors to Gr00t state format.

    Args:
        states: Dictionary mapping state names to state tensors.
            Expected shape: (B, D) with dtype float32.
        temporal_stack: Optional dictionary of previous states for temporal stacking.
            If provided, states will be stacked along temporal dimension.

    Returns:
        Dictionary mapping state names to numpy arrays of shape (B, T, D)
        with dtype float32.

    Example:
        >>> states = {"joint_positions": torch.zeros(4, 7)}
        >>> groot_state = isaaclab_state_to_groot(states)
        >>> groot_state["joint_positions"].shape
        (4, 1, 7)
    """
    result = {}

    for state_name, state_tensor in states.items():
        # Ensure tensor is on CPU and convert to numpy
        if state_tensor.is_cuda:
            state_tensor = state_tensor.cpu()

        state_np = state_tensor.numpy().astype(np.float32)

        # Ensure shape is (B, D)
        if state_np.ndim == 1:
            # Single environment (D,) -> (1, D)
            state_np = state_np[np.newaxis, ...]

        # Stack temporally if previous states are provided
        if temporal_stack is not None and state_name in temporal_stack:
            prev_states = temporal_stack[state_name]
            all_states = [s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in prev_states]
            all_states.append(state_np)
            # Stack along new temporal dimension: (B, T, D)
            state_np = np.stack(all_states, axis=1)
        else:
            # Add temporal dimension: (B, D) -> (B, 1, D)
            state_np = state_np[:, np.newaxis, ...]

        result[state_name] = state_np

    return result


def format_language_instruction(
    instruction: str | list[str],
    num_envs: int,
    temporal_horizon: int = 1,
    language_key: str = "annotation.human.task_description",
) -> dict[str, list[str] | tuple[str, ...]]:
    """Format language instruction for Gr00t.

    Args:
        instruction: Task instruction string or list of strings (one per env).
        num_envs: Number of parallel environments.
        temporal_horizon: Temporal horizon for the instruction (unused, kept for API compat).
        language_key: The key name for the language modality (default: "annotation.human.task_description").

    Returns:
        Dictionary with language_key mapping to tuple of strings of shape (B,).

    Example:
        >>> lang = format_language_instruction("pick up the cube", num_envs=4)
        >>> len(lang["annotation.human.task_description"])
        4
    """
    if isinstance(instruction, str):
        # Same instruction for all environments
        instructions = tuple([instruction] * num_envs)
    else:
        # Per-environment instructions
        if len(instruction) != num_envs:
            raise ValueError(f"Expected {num_envs} instructions, got {len(instruction)}")
        instructions = tuple(instruction)

    return {language_key: instructions}


def groot_action_to_isaaclab(
    action: dict[str, np.ndarray],
    action_keys: list[str] | None = None,
    action_horizon_idx: int = 0,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Convert Gr00t action output to IsaacLab action tensor.

    Args:
        action: Dictionary of action arrays from Gr00t.
            Each array has shape (B, T_action, D) with dtype float32.
        action_keys: Ordered list of action keys to concatenate.
            If None, uses all keys in the order they appear in the dict.
        action_horizon_idx: Which action in the horizon to use (0 = first/current).
        device: Target torch device.

    Returns:
        Action tensor of shape (B, total_action_dim) ready for env.step().

    Example:
        >>> action = {"arm": np.zeros((4, 16, 7)), "gripper": np.zeros((4, 16, 1))}
        >>> isaaclab_action = groot_action_to_isaaclab(action, device="cuda:0")
        >>> isaaclab_action.shape
        torch.Size([4, 8])
    """
    if action_keys is None:
        action_keys = list(action.keys())

    action_tensors = []
    for key in action_keys:
        if key not in action:
            raise KeyError(f"Action key '{key}' not found in action dict. Available: {list(action.keys())}")

        arr = action[key]

        # Handle different array shapes
        if arr.ndim == 3:
            # (B, T_action, D) -> select specific horizon step
            selected = arr[:, action_horizon_idx, :]  # (B, D)
        elif arr.ndim == 2:
            # (B, D) -> use as is
            selected = arr
        else:
            raise ValueError(f"Expected 2D or 3D array for action '{key}', got shape {arr.shape}")

        action_tensors.append(torch.from_numpy(selected).to(device))

    return torch.cat(action_tensors, dim=-1)


def build_groot_observation(
    video: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    language: dict[str, list[list[str]]],
) -> dict[str, Any]:
    """Build a complete Gr00t observation dictionary with flat keys.

    Gr00t expects flat keys like 'video.camera_name', 'state.joint_position',
    not nested dictionaries.

    Args:
        video: Video observations from isaaclab_images_to_groot().
        state: State observations from isaaclab_state_to_groot().
        language: Language instructions from format_language_instruction().

    Returns:
        Complete observation dictionary with flat keys ready for PolicyClient.get_action().

    Example:
        >>> video = {"ego_view": np.zeros((4, 1, 224, 224, 3), dtype=np.uint8)}
        >>> state = {"joint_position": np.zeros((4, 1, 7), dtype=np.float32)}
        >>> language = {"task": [["pick up cube"]] * 4}
        >>> obs = build_groot_observation(video, state, language)
        >>> "video.ego_view" in obs
        True
    """
    result = {}

    # Flatten video keys: {"cam": arr} -> {"video.cam": arr}
    for key, value in video.items():
        result[f"video.{key}"] = value

    # Flatten state keys: {"joints": arr} -> {"state.joints": arr}
    for key, value in state.items():
        result[f"state.{key}"] = value

    # Language keys stay as-is (e.g., "task" -> "task")
    for key, value in language.items():
        result[key] = value

    return result
