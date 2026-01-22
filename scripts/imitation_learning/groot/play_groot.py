# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run Isaac-Gr00t VLA policy inference in IsaacLab.

This script connects to a running Isaac-Gr00t inference server and
uses it to control a robot in an IsaacLab simulation environment.

Usage:
    # First, start the Isaac-Gr00t server (in another terminal):
    python -m gr00t.eval.run_gr00t_server --model_path /path/to/model --port 5555

    # Then run this script:
    python scripts/imitation_learning/groot/play_groot.py \
        --task Isaac-Groot-Lift-Cube-Franka-v0 \
        --language_instruction "pick up the cube" \
        --server_host localhost \
        --server_port 5555

Args:
    task: Name of the IsaacLab environment.
    server_host: Isaac-Gr00t server hostname.
    server_port: Isaac-Gr00t server port.
    language_instruction: Task instruction for the VLA policy.
    num_envs: Number of parallel environments.
    horizon: Maximum episode length in steps.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Run Isaac-Gr00t VLA policy inference.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Isaac-Groot-Lift-Cube-Franka-v0", help="Name of the task.")
parser.add_argument("--server_host", type=str, default="localhost", help="Isaac-Gr00t server hostname.")
parser.add_argument("--server_port", type=int, default=5555, help="Isaac-Gr00t server port.")
parser.add_argument(
    "--language_instruction", type=str, default="pick up the cube", help="Task instruction for VLA policy."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--horizon", type=int, default=1000, help="Maximum episode length in steps.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--video_horizon", type=int, default=1, help="Number of video frames to stack.")
parser.add_argument("--state_horizon", type=int, default=1, help="Number of state steps to stack.")
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio for Pink IK.")
parser.add_argument("--test_env", default=False, action="store_true", help="Test environment only, no server connection.")
parser.add_argument(
    "--groot_video_keys", type=str, nargs="+", default=None,
    help="Gr00t model video key names (e.g., 'ego_view_bg_crop_pad_res256_freq20'). Must match model's training config."
)
parser.add_argument(
    "--groot_state_keys", type=str, nargs="+", default=None,
    help="Gr00t model state key names (e.g., 'joint_position'). Must match model's training config."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab
    # and not the one installed by Isaac Sim. Required by Pink IK controllers.
    import pinocchio  # noqa: F401

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import random
import torch

# Import Gr00t environments to register them
import isaaclab_tasks.manager_based.manipulation.groot.config.franka  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.groot.config.gr1t2  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.groot.config.g1  # noqa: F401

from isaaclab.policy_clients import Gr00tPolicyClient
from isaaclab_tasks.utils import parse_env_cfg


def run_inference(
    env: gym.Env,
    client: Gr00tPolicyClient,
    language_instruction: str,
    horizon: int,
) -> tuple[bool, dict]:
    """Run a single episode of inference.

    Args:
        env: The IsaacLab environment.
        client: The Gr00t policy client.
        language_instruction: Task instruction.
        horizon: Maximum episode length.

    Returns:
        Tuple of (success, trajectory_info).
    """
    # Reset environment and client
    obs_dict, _ = env.reset()
    client.reset()

    trajectory = {"observations": [], "actions": [], "rewards": []}
    total_reward = 0.0

    for step in range(horizon):
        # Get action from Gr00t server
        try:
            action = client.get_action(obs_dict, language_instruction)
        except Exception as e:
            print(f"[ERROR] Failed to get action from server: {e}")
            break

        # Apply action to environment
        obs_dict, reward, terminated, truncated, info = env.step(action)

        # Record trajectory
        trajectory["actions"].append(action.cpu().numpy())
        trajectory["rewards"].append(reward.cpu().numpy())
        total_reward += reward.sum().item()

        # Check for episode end
        if terminated.any() or truncated.any():
            # Handle per-env resets
            reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids) > 0:
                client.reset(reset_ids)

            # If all envs are done, break
            if terminated.all() or truncated.all():
                break

    # Check success (simplified - assumes success term exists)
    success = False
    if hasattr(env.unwrapped, "cfg") and hasattr(env.unwrapped.cfg, "terminations"):
        if hasattr(env.unwrapped.cfg.terminations, "success"):
            try:
                success_term = env.unwrapped.cfg.terminations.success
                if success_term is not None:
                    success = bool(success_term.func(env.unwrapped, **success_term.params).any())
            except Exception:
                pass

    trajectory_info = {
        "total_reward": total_reward,
        "num_steps": step + 1,
        "success": success,
    }

    return success, trajectory_info


def main():
    """Main function to run Gr00t policy inference."""
    # Set random seeds
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Ensure observations are in dictionary mode for Gr00t
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.concatenate_terms = False

    # Disable recorders for inference
    env_cfg.recorders = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print("[INFO] Environment created successfully!")

    # Test environment mode - just run with idle/zero actions
    if args_cli.test_env:
        print("[INFO] Test mode: Running environment without Gr00t server...")
        obs_dict, _ = env.reset()
        print(f"[INFO] Observation keys: {list(obs_dict['policy'].keys()) if 'policy' in obs_dict else list(obs_dict.keys())}")

        # Get idle action if available, otherwise use zeros
        idle_action = None #getattr(env_cfg, "idle_action", None)
        if idle_action is not None:
            action = idle_action.unsqueeze(2).to(env.device)
            print(f"[INFO] Using idle action with shape {action.shape}")
        else:
            action = torch.zeros(args_cli.num_envs, env.action_space.shape[1], device=env.device)
            print(f"[INFO] Using zero action with shape {action.shape}")

        step = 0
        try:
            while simulation_app.is_running():
                obs_dict, reward, terminated, truncated, info = env.step(action)
                step += 1
                if step % 100 == 0:
                    print(f"[INFO] Step {step}, reward: {reward.mean().item():.4f}")
                if terminated.any() or truncated.any():
                    print(f"[INFO] Episode ended at step {step}, resetting...")
                    obs_dict, _ = env.reset()
                    step = 0
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            env.close()
        return

    # Get Gr00t client configuration from environment config
    camera_keys = getattr(env_cfg, "camera_keys", ["robot_pov_cam"])
    state_keys = getattr(env_cfg, "state_keys", ["left_arm", "right_arm", "left_hand", "right_hand", "waist"])
    action_keys = getattr(env_cfg, "action_keys", None)

    # Get Gr00t model key mappings - CLI args override env config
    groot_video_keys = args_cli.groot_video_keys or getattr(env_cfg, "groot_video_keys", camera_keys)
    groot_state_keys = args_cli.groot_state_keys or getattr(env_cfg, "groot_state_keys", state_keys)

    print(f"[INFO] Connecting to Gr00t server at {args_cli.server_host}:{args_cli.server_port}")
    print(f"[INFO] IsaacLab camera keys: {camera_keys}")
    print(f"[INFO] IsaacLab state keys: {state_keys}")
    print(f"[INFO] Gr00t video keys: {groot_video_keys}")
    print(f"[INFO] Gr00t state keys: {groot_state_keys}")
    print(f"[INFO] Action keys: {action_keys}")

    # Create Gr00t client
    client = Gr00tPolicyClient(
        host=args_cli.server_host,
        port=args_cli.server_port,
        num_envs=args_cli.num_envs,
        video_keys=camera_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        groot_video_keys=groot_video_keys,
        groot_state_keys=groot_state_keys,
        video_horizon=args_cli.video_horizon,
        state_horizon=args_cli.state_horizon,
        device=str(env.device),
        auto_fetch_config=True,
    )

    print(f"[INFO] Key mapping - video: {dict(zip(camera_keys, groot_video_keys))}")
    print(f"[INFO] Key mapping - state: {dict(zip(state_keys, groot_state_keys))}")

    # Test connection
    if not client.connect():
        print("[ERROR] Failed to connect to Gr00t server. Is it running?")
        print(f"[ERROR] Expected server at tcp://{args_cli.server_host}:{args_cli.server_port}")
        env.close()
        return

    print("[INFO] Successfully connected to Gr00t server!")
    print(f"[INFO] Language instruction: '{args_cli.language_instruction}'")
    print(f"[INFO] Running inference for up to {args_cli.horizon} steps...")

    # Run inference loop
    episode = 0
    try:
        while simulation_app.is_running():
            episode += 1
            print(f"\n[INFO] Starting episode {episode}")

            success, info = run_inference(
                env=env,
                client=client,
                language_instruction=args_cli.language_instruction,
                horizon=args_cli.horizon,
            )

            print(f"[INFO] Episode {episode} finished:")
            print(f"  - Steps: {info['num_steps']}")
            print(f"  - Total reward: {info['total_reward']:.2f}")
            print(f"  - Success: {success}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        print("[INFO] Closing connections...")
        client.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
