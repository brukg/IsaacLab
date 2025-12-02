# Vision-Based Locomotion and Navigation Guide

Complete documentation for vision-enabled humanoid robot control in IsaacLab.

**Quick Navigation:**
- [H1 Locomotion Architectures](#h1-locomotion-architectures)
- [H1 Navigation Environments](#h1-navigation-environments)
- [G1 Vision Support](#g1-vision-support)
- [Training Commands](#training-commands)
- [Environment Registry](#environment-registry)
- [Changelog](#changelog)

---

# H1 Locomotion Architectures

## Overview

| Architecture | Vision | RNN | IMU | Parameters | GPU Memory | Use Case |
|-------------|--------|-----|-----|------------|------------|----------|
| Standard PPO | No | No | No | ~403K | ~8 GB | Flat terrain, fast training |
| Depth CNN | Yes | No | No | ~1.84M | ~18 GB | Obstacle avoidance, reactive |
| Depth CNN + RNN | Yes | Yes | No | ~3.15M | ~24 GB | Complex obstacles, predictive |
| Depth CNN + IMU | Yes | No | Yes | ~1.84M | ~18 GB | Sim-to-real preparation |
| Depth CNN + RNN + IMU | Yes | Yes | Yes | ~3.15M | ~24 GB | Full stack, most realistic |

---

## Standard PPO (No Vision)

**Environment:** `Isaac-Velocity-Rough-H1-v0`

### Architecture
```
Observations (69) → Actor MLP [512→256→128] → Actions (19)
                 ↓
                 Critic MLP [512→256→128] → Value
```

### Observations (69 dims)
- Base linear velocity (3), angular velocity (3)
- Projected gravity (3)
- Velocity commands (3)
- Joint positions (19), velocities (19)
- Previous actions (19)

**Pros**: Fast training, low GPU usage, good for simple terrain
**Cons**: Cannot see obstacles, reactive only

---

## Depth CNN (Vision)

**Environments:**
- `Isaac-Velocity-Rough-H1-Vision-v0` (Ground truth)
- `Isaac-Velocity-Rough-H1-Vision-IMU-v0` (IMU sensors)

### Architecture
```
Actor:
  Proprio (69) → MLP [512→256→128] ──┐
                                      ├→ Concat (256) → Action Head → Actions
  Depth (1590) → CNN Backbone → 128 ─┘

Critic:
  Proprio (69) + Depth (1590) + Height_scan (160) → Depth CNN → MLP → Value
```

### Sensors

**Depth Camera:**
- Resolution: 53 × 30 pixels (1590 values)
- Range: 0.3m - 5.0m (extended from legged-loco's 2.0m)
- Attachment: torso_link (forward-facing)
- Update: Every physics step

**Height Scanner:**
- Grid: 16 × 10 pattern (160 values)
- Coverage: 1.6m × 1.0m area
- Attachment: torso_link (downward-facing)
- Use: Terrain elevation (critic only)

**IMU Sensor (Optional):**
- Gyroscope: Angular velocity (replaces base_ang_vel)
- Accelerometer: Projected gravity (replaces projected_gravity)
- Linear velocity: Still ground truth (simulates state estimator)

### Observations
- **Ground truth**: Actor 1659 dims, Critic 1915 dims
- **IMU variant**: Same dimensions, different sources

**Parameters**: ~1.84M
**Pros**: Obstacle detection, forward-looking awareness
**Cons**: No temporal memory, reactive only

---

## Depth CNN + RNN (Vision + Memory)

**Environments:**
- `Isaac-Velocity-Rough-H1-Vision-RNN-v0` (Ground truth)
- `Isaac-Velocity-Rough-H1-Vision-IMU-RNN-v0` (IMU sensors)

### Architecture
```
Actor:
  Proprio (69) → MLP [512→256→128] ──┐
                                      ├→ Concat (256) → LSTM (256) → Action Head
  Depth (1590) → CNN Backbone → 128 ─┘

Critic:
  Proprio + Depth + Height (1915) → Encoder MLP [1915→512→256]
                                         ↓
                                       LSTM (256) → MLP [256→512→256→128→1]
```

### RNN Configuration
- Type: LSTM (or GRU)
- Hidden size: 256
- Layers: 1
- Critic encoder: Reduces 1915 → 256 dims before RNN

**Parameters**: ~3.15M
**Pros**: Temporal context, predictive behavior, smoother trajectories
**Cons**: Slower training, higher GPU usage

---

## Terrain Configuration

Enhanced from legged-loco with increased obstacle density:

- Pyramid stairs: 15% (forward + 15% inverted)
- Random boxes: 35% (increased from 20%)
- Random rough: 5%
- Slopes: 5% (forward + 5% inverted)
- Giant obstacles: 40% (1.5m tall, 900 count)

**Total obstacle coverage:** ~75% (vs legged-loco's 40%)

---

## H1 Locomotion Environments (12 total)

### Standard (4)
1. Isaac-Velocity-Rough-H1-v0
2. Isaac-Velocity-Rough-H1-Play-v0
3. Isaac-Velocity-Flat-H1-v0
4. Isaac-Velocity-Flat-H1-Play-v0

### Vision - Ground Truth (4)
5. Isaac-Velocity-Rough-H1-Vision-v0
6. Isaac-Velocity-Rough-H1-Vision-Play-v0
7. Isaac-Velocity-Rough-H1-Vision-RNN-v0
8. Isaac-Velocity-Rough-H1-Vision-RNN-Play-v0

### Vision - IMU (4)
9. Isaac-Velocity-Rough-H1-Vision-IMU-v0
10. Isaac-Velocity-Rough-H1-Vision-IMU-Play-v0
11. Isaac-Velocity-Rough-H1-Vision-IMU-RNN-v0
12. Isaac-Velocity-Rough-H1-Vision-IMU-RNN-Play-v0

---

# H1 Navigation Environments

## Hierarchical Control Architecture

H1 navigation uses **two-level control**:
- **High-level policy** (10Hz): Goal-directed navigation using vision
- **Low-level policy** (50Hz): Pre-trained locomotion controller
- **Benefit**: Separation of planning vs execution

---

## Navigation Environment Variants (12 total)

### Base Navigation (2)
- Isaac-Navigation-Flat-H1-v0
- Isaac-Navigation-Flat-H1-Play-v0

**Observations**: 13 dims (velocities, gravity, pose_command)
**Use case**: Baseline without vision, open terrain

### Depth Camera Only (4)
- Isaac-Navigation-Flat-H1-Depth-v0
- Isaac-Navigation-Flat-H1-Depth-Play-v0
- Isaac-Navigation-Flat-H1-Depth-RNN-v0
- Isaac-Navigation-Flat-H1-Depth-RNN-Play-v0

**Observations**: Proprio (13) + Depth (1590) = 1603 dims
**Use case**: Forward obstacle avoidance

### Height Scanner Only (2)
- Isaac-Navigation-Flat-H1-Scanner-v0
- Isaac-Navigation-Flat-H1-Scanner-Play-v0

**Observations**: Proprio (13) + Scanner (160) = 173 dims
**Use case**: Terrain-aware navigation

### Depth + Scanner Combined (4)
- Isaac-Navigation-Flat-H1-DepthScanner-v0
- Isaac-Navigation-Flat-H1-DepthScanner-Play-v0
- Isaac-Navigation-Flat-H1-DepthScanner-RNN-v0
- Isaac-Navigation-Flat-H1-DepthScanner-RNN-Play-v0

**Observations**: Actor 1603 dims, Critic 1763 dims (includes scanner)
**Use case**: Full perception stack

---

## Navigation Network Architectures

### Depth Camera Only (~1.2M params)
```
Actor: Proprio (13) → MLP + Depth (1590) → CNN → High-level Commands
Critic: Proprio + Depth → Depth CNN → Value
```

### Depth Camera + RNN (~2.5M params)
```
Actor: Proprio + Depth → Depth CNN → LSTM (256) → Commands
Critic: Proprio + Depth → Encoder → LSTM (256) → Value
```

### Height Scanner Only (~300K params)
```
Actor/Critic: Proprio (13) + Scanner (160) → MLP → Output
```

### Full Stack (~2.7M params)
```
Actor: Proprio + Depth → Depth CNN → LSTM → Commands
Critic: Proprio + Depth + Scanner → Encoder → LSTM → Value
```

---

## Navigation Configuration

- **Task**: Reach target pose (x, y, heading)
- **Goal range**: x,y ∈ [-5, 5]m, heading ∈ [-π, π]
- **Episode length**: 10 seconds per goal
- **Rewards**: Position tracking (coarse + fine), orientation tracking
- **Control frequency**: High-level 10Hz, Low-level 50Hz

### Prerequisites

Navigation requires a pre-trained H1 flat locomotion policy.

**Step 1: Train the low-level policy**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-H1-v0 \
  --headless
```

**Step 2: Export the policy**
After training completes, the policy is automatically saved to:
```
logs/rsl_rl/h1_flat/exported/policy.pt
```

**Step 3: Use the policy**

The navigation environments will automatically use the local policy if it exists at:
```
logs/rsl_rl/h1_flat/exported/policy.pt
```

**Override via command line** (recommended):
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-Depth-v0 \
  --low_level_policy logs/rsl_rl/your_experiment/exported/policy.pt \
  --headless
```

**Override in config file**:
Edit line 89 in `navigation_env_cfg.py`:
```python
policy_path="logs/rsl_rl/your_experiment/exported/policy.pt"
```

---

## Locomotion vs Navigation

| Feature | Locomotion (Velocity) | Navigation (Goal-directed) |
|---------|----------------------|---------------------------|
| Control | Direct velocity commands | Hierarchical (goal → velocity) |
| Task | Walk at commanded velocity | Reach target pose |
| Episode | 20-40 seconds | 10 seconds per goal |
| Low-level | Learned jointly | Pre-trained, frozen |
| Frequency | 50 Hz | 10 Hz (high-level) |
| Observations | Full state | Goal-relative |

---

# G1 Vision Support

## G1 Locomotion Environments (6 total)

### Standard (4)
1. Isaac-Velocity-Rough-G1-v0 (PPO, TD3, FastTD3)
2. Isaac-Velocity-Rough-G1-Play-v0
3. Isaac-Velocity-Flat-G1-v0 (PPO, TD3, FastTD3)
4. Isaac-Velocity-Flat-G1-Play-v0

### Vision (2)
5. Isaac-Velocity-Rough-G1-Vision-v0
6. Isaac-Velocity-Rough-G1-Vision-Play-v0

## G1 Configuration

- **Depth Camera**: 53×30 resolution, 0.3-5.0m range
- **Height Scanner**: 16×10 grid pattern
- **Proprioception**: 123 dims (37 joints × 2 + velocities + commands)
- **Agent**: ActorCriticDepthCNN
- **TD3 Support**: Available for rough and flat terrain

---

# Training Commands

## H1 Locomotion

### Standard PPO
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-v0 --headless
```

### Depth CNN (Ground Truth)
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-Vision-v0 --headless
```

### Depth CNN + RNN
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-Vision-RNN-v0 --headless
```

### Depth CNN (IMU)
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-Vision-IMU-v0 --headless
```

### Depth CNN + RNN (IMU)
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-Vision-IMU-RNN-v0 --headless
```

---

## H1 Navigation

### Base Navigation
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-v0 \
  --low_level_policy logs/rsl_rl/h1_flat/exported/policy.pt \
  --headless
```

### Depth Camera Only
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-Depth-v0 \
  --low_level_policy logs/rsl_rl/h1_flat/exported/policy.pt \
  --headless
```

### Depth + RNN
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-Depth-RNN-v0 \
  --low_level_policy logs/rsl_rl/h1_flat/exported/policy.pt \
  --headless
```

### Height Scanner Only
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-Scanner-v0 \
  --low_level_policy logs/rsl_rl/h1_flat/exported/policy.pt \
  --headless
```

### Depth + Scanner
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-DepthScanner-v0 \
  --low_level_policy logs/rsl_rl/h1_flat/exported/policy.pt \
  --headless
```

### Depth + Scanner + RNN (Full Stack)
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-DepthScanner-RNN-v0 \
  --low_level_policy logs/rsl_rl/h1_flat/exported/policy.pt \
  --headless
```

---

## G1 Locomotion

### Depth CNN (PPO)
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-Vision-v0 --headless
```

### TD3
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-v0 --agent rsl_rl_td3_cfg --headless
```

### FastTD3
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-v0 --agent rsl_rl_fast_td3_cfg --headless
```

---

## Inference/Play

Replace training task with `-Play-v0` variant:
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-H1-Vision-Play-v0 --num_envs 50
```

---

# Environment Registry

## Complete Environment List (30 total)

### H1 Humanoid Locomotion (12)
- Standard: 4 (rough, flat + play variants)
- Vision (ground truth): 4 (depth CNN, depth CNN + RNN + play)
- Vision (IMU): 4 (depth CNN, depth CNN + RNN + play)

### H1 Humanoid Navigation (12)
- Base: 2
- Depth only: 4 (with/without RNN + play)
- Scanner only: 2
- Depth + Scanner: 4 (with/without RNN + play)

### G1 Humanoid Locomotion (6)
- Standard: 4 (rough, flat + play, with TD3/FastTD3)
- Vision: 2 (depth CNN + play)

---

# GPU Memory Requirements

| Variant | Num Envs | GPU Memory | Recommendation |
|---------|----------|------------|----------------|
| Standard PPO | 4096 | ~8 GB | RTX 3070+ |
| Depth CNN | 4096 | ~18 GB | RTX 3090+ |
| Depth CNN + RNN | 4096 | ~24 GB | RTX 4090 / A100 |
| Navigation (Depth) | 4096 | ~18 GB | RTX 3090+ |
| Navigation (Scanner) | 4096 | ~10 GB | RTX 3080+ |

**To reduce GPU usage:**
```python
# In environment config
self.scene.num_envs = 2048  # Halves memory
obs_depth_shape = (40, 24)  # Reduces depth resolution
```

---

# Changelog

## Added Features

### H1 Navigation Environments (12 total)
- Hierarchical control (high-level 10Hz, low-level 50Hz)
- Depth camera only variants (4)
- Height scanner only variants (2)
- Combined depth + scanner variants (4)
- Base navigation without vision (2)
- RNN support for temporal memory

### H1 Locomotion Environments (8 vision variants)
- Ground truth observation variants (4)
- IMU sensor variants (4)
- RNN/LSTM support for temporal processing
- Enhanced terrain (75% obstacle coverage)
- Extended depth range (5.0m vs legged-loco's 2.0m)

### G1 Locomotion Environments (2 vision variants)
- Depth CNN support
- TD3/FastTD3 off-policy learning
- Extended depth camera range

### TD3/FastTD3 Support
- Off-policy reinforcement learning
- Added to 10+ environments
- Replay buffer, twin critics, policy delay

---

## Bug Fixes

1. **RNN Integration**: Fixed 6 bugs in ActorCriticDepthCNNRecurrent
   - Memory initialization parameter names
   - Hidden state attribute names
   - Dimension mismatches (encoder, critic MLP)

2. **ONNX Export**: Fixed depth CNN + RNN export
   - Observation size calculation
   - Encoding step in forward methods

3. **TD3 Checkpoint Loading**: Added strict=False for missing normalizers

4. **Training Script**: Added directory existence check before checkpoint loading

---

## Comparison with Legged-Loco

**Inspired by:**
- Depth camera resolution (53×30)
- Height scanner configuration
- Terrain generation approach

**Enhanced/Different:**
- Novel hybrids (H1 + G1 vision, navigation support)
- Extended depth range (5.0m vs 2.0m)
- RNN/LSTM temporal memory
- IMU sensor variants
- Higher obstacle density (75% vs 40%)
- Hierarchical navigation framework
- TD3/FastTD3 off-policy learning

---

## File Locations

### H1 Locomotion
- Environments: `isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_vision_env_cfg.py`
- Agents: `config/h1/agents/rsl_rl_depth_cnn_cfg.py`, `rsl_rl_depth_cnn_rnn_cfg.py`
- Registry: `config/h1/__init__.py`

### H1 Navigation
- Environments: `isaaclab_tasks/manager_based/navigation/config/h1/navigation_env_cfg.py`
- Agents: `config/h1/agents/rsl_rl_depth_cfg.py`, `rsl_rl_scanner_cfg.py`, etc.
- Registry: `config/h1/__init__.py`

### G1 Locomotion
- Environments: `isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_vision_env_cfg.py`
- Agents: `config/g1/agents/rsl_rl_depth_cnn_cfg.py`, `rsl_rl_td3_cfg.py`
- Registry: `config/g1/__init__.py`

### Infrastructure
- TD3 Runner: `source/isaaclab_rl/rsl_rl/runners/off_policy_runner.py`
- TD3 Algorithm: `rsl_rl/rsl_rl/algorithms/td3.py`
- Depth CNN: `rsl_rl/rsl_rl/modules/actor_critic_depth_cnn.py`
- Exporter: `source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py`

---

**Contributors:** Implementation based on legged-loco foundation with significant enhancements
**Date:** January 2025
**IsaacLab Version:** 1.0.0+
**License:** BSD-3-Clause
