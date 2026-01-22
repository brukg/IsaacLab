# Isaac-Gr00t Integration

This directory contains scripts for running Isaac-Gr00t VLA (Vision-Language-Action) policy inference in IsaacLab.

## Overview

The integration allows you to:
- Run IsaacLab simulation environments
- Connect to an Isaac-Gr00t inference server via ZMQ
- Use VLA policies to control robots with camera observations and language instructions

## Prerequisites

1. **Isaac-Gr00t Server**: You need a running Isaac-Gr00t inference server. See the [Isaac-Gr00t documentation](https://github.com/nvidia/Isaac-Gr00t) for setup instructions.

2. **Dependencies**: Ensure you have the required packages:
   ```bash
   pip install pyzmq msgpack
   ```

## Quick Start

### Step 1: Start the Isaac-Gr00t Server

In a separate terminal, start the Gr00t inference server:

```bash
# Using a trained model
python -m gr00t.eval.run_gr00t_server \
    --model_path /path/to/checkpoint \
    --embodiment_tag new_embodiment \
    --port 5555

# Or using a replay policy for testing
python -m gr00t.eval.run_gr00t_server \
    --dataset_path /path/to/lerobot_dataset \
    --embodiment_tag libero_panda \
    --port 5555
```

### Step 2: Run IsaacLab Inference

```bash
# Franka lift task
python scripts/imitation_learning/groot/play_groot.py \
    --task Isaac-Groot-Lift-Cube-Franka-v0 \
    --language_instruction "pick up the cube" \
    --server_host localhost \
    --server_port 5555

# GR1T2 humanoid pick-place task
python scripts/imitation_learning/groot/play_groot.py \
    --task Isaac-Groot-PickPlace-GR1T2-v0 \
    --language_instruction "pick up the steering wheel" \
    --server_host localhost \
    --server_port 5555
```

## Available Environments

| Environment ID | Robot | Task |
|---------------|-------|------|
| `Isaac-Groot-Lift-Cube-Franka-v0` | Franka Panda | Lift cube |
| `Isaac-Groot-Lift-Cube-Franka-Play-v0` | Franka Panda | Lift cube (multi-env) |
| `Isaac-Groot-PickPlace-GR1T2-v0` | GR1T2 Humanoid | Pick and place |
| `Isaac-Groot-PickPlace-GR1T2-Play-v0` | GR1T2 Humanoid | Pick and place (multi-env) |

## Architecture

```
┌─────────────────┐     ZMQ (REQ-REP)     ┌─────────────────┐
│   IsaacLab      │ ◄──────────────────► │  Isaac-Gr00t    │
│   Simulation    │    msgpack + numpy    │  Inference      │
│                 │                       │  Server         │
│  - Observations │ ────────────────────► │  - VLA Model    │
│  - Actions      │ ◄──────────────────── │  - PolicyServer │
└─────────────────┘                       └─────────────────┘
```

### Observation Format

The environments provide observations in the format expected by Gr00t:

```python
observation = {
    "video": {
        "wrist_camera": np.ndarray(B, T, H, W, 3),  # uint8, wrist view
        "scene_camera": np.ndarray(B, T, H, W, 3),  # uint8, external view
    },
    "state": {
        "joint_positions": np.ndarray(B, T, D),    # float32
    },
    "language": {
        "task": [["pick up the cube"]] * B,        # list[list[str]]
    }
}
```

### Action Format

Gr00t returns action chunks that are converted to IsaacLab format:

```python
# Gr00t output
action = {
    "joint_positions": np.ndarray(B, T_action, D),  # Action chunks
}

# Converted to IsaacLab
action_tensor = torch.Tensor(B, action_dim)  # First action in horizon
```

## Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Environment name | `Isaac-Groot-Lift-Cube-Franka-v0` |
| `--server_host` | Gr00t server hostname | `localhost` |
| `--server_port` | Gr00t server port | `5555` |
| `--language_instruction` | Task instruction | `"pick up the cube"` |
| `--num_envs` | Number of parallel environments | `1` |
| `--horizon` | Maximum episode length | `1000` |
| `--video_horizon` | Video temporal horizon | `1` |
| `--state_horizon` | State temporal horizon | `1` |

### Creating Custom Environments

To create a new Gr00t-compatible environment:

1. Create a new environment config inheriting from `Gr00tEnvCfg`:

```python
from isaaclab_tasks.manager_based.manipulation.groot.groot_env_cfg import Gr00tEnvCfg

@configclass
class MyGr00tEnvCfg(Gr00tEnvCfg):
    # Configure scene, robot, cameras
    scene: MySceneCfg = MySceneCfg(...)

    # Specify observation keys for Gr00t client
    camera_keys: list[str] = ["wrist_camera", "scene_camera"]
    state_keys: list[str] = ["joint_positions"]
    action_keys: list[str] = ["arm_action", "gripper_action"]
```

2. Register with gymnasium:

```python
import gymnasium as gym

gym.register(
    id="Isaac-Groot-MyTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "my_module:MyGr00tEnvCfg"},
)
```

## Troubleshooting

### Connection Failed

If you see "Failed to connect to Gr00t server":
1. Ensure the Gr00t server is running
2. Check the host and port match
3. Verify no firewall is blocking the connection

### Observation Shape Mismatch

If actions fail due to shape mismatches:
1. Check the modality config from the server matches your environment
2. Verify camera resolution (default 224x224)
3. Check temporal horizon settings

### Low Inference Speed

For better performance:
1. Use GPU for the Gr00t server (`--device cuda`)
2. Reduce the number of environments if memory-limited
3. Enable action chunking to reduce server calls
