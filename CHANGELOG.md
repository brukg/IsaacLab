# IsaacLab Locomotion Extensions - Changelog

## [Unreleased] - 2025-01-XX

This release adds vision-based locomotion, off-policy reinforcement learning (TD3/FastTD3), and enhanced configurations for humanoid robots.

---

## Added Features

### 1. H1 Navigation Environments with Vision Support

Complete suite of vision-based navigation environments for H1 humanoid robot using hierarchical control.

#### Hierarchical Control Architecture
- **High-level policy**: Goal-directed navigation (10Hz) using vision
- **Low-level policy**: Pre-trained locomotion controller (50Hz)
- **Separation of concerns**: Planning vs execution

#### Environment Variants (12 total)

**Base Navigation (2)**:
- Isaac-Navigation-Flat-H1-v0
- Isaac-Navigation-Flat-H1-Play-v0

**Depth Camera Only (4)**:
- Isaac-Navigation-Flat-H1-Depth-v0
- Isaac-Navigation-Flat-H1-Depth-Play-v0
- Isaac-Navigation-Flat-H1-Depth-RNN-v0 (with LSTM)
- Isaac-Navigation-Flat-H1-Depth-RNN-Play-v0

**Height Scanner Only (2)**:
- Isaac-Navigation-Flat-H1-Scanner-v0
- Isaac-Navigation-Flat-H1-Scanner-Play-v0

**Depth + Scanner Combined (4)**:
- Isaac-Navigation-Flat-H1-DepthScanner-v0
- Isaac-Navigation-Flat-H1-DepthScanner-Play-v0
- Isaac-Navigation-Flat-H1-DepthScanner-RNN-v0 (with LSTM)
- Isaac-Navigation-Flat-H1-DepthScanner-RNN-Play-v0

#### Sensor Configurations

**Depth Camera**:
- Resolution: 53 × 30 pixels (1590 values)
- Range: 0.3m - 5.0m
- Attachment: torso_link (forward-facing)
- Pattern: Pinhole camera (focal_length=1.93)

**Height Scanner**:
- Grid: 16 × 10 pattern (160 values)
- Coverage: 1.6m × 1.0m area
- Attachment: torso_link (downward-facing)
- Use case: Terrain elevation awareness

#### Network Architectures

**Depth Camera Only**: ~1.2M parameters
```
Actor: Proprio (13) → MLP + Depth (1590) → CNN → Action
Critic: Proprio + Depth → Depth CNN → Value
```

**Depth Camera + RNN**: ~2.5M parameters
```
Actor: Proprio + Depth → Depth CNN → LSTM (256) → Action
Critic: Proprio + Depth → Encoder → LSTM (256) → Value
```

**Height Scanner Only**: ~300K parameters
```
Actor/Critic: Proprio (13) + Scanner (160) → MLP → Output
```

**Full Stack (Depth + Scanner + RNN)**: ~2.7M parameters
```
Actor: Proprio + Depth → Depth CNN → LSTM → Action
Critic: Proprio + Depth + Scanner → Encoder → LSTM → Value
```

#### Training Configuration
- **Task**: Goal-directed navigation (pose commands)
- **Episode length**: 10 seconds per goal
- **Goal range**: x,y ∈ [-5, 5]m, heading ∈ [-π, π]
- **Rewards**: Position tracking (coarse + fine-grained), orientation tracking
- **Low-level decimation**: 4× (200Hz → 50Hz)
- **High-level decimation**: 10× (50Hz → 10Hz)

#### Observation Groups
- **Base**: Proprio (13 dims) - velocities, gravity, pose command
- **Depth**: Separate observation group for depth CNN processing
- **Scanner**: Separate observation group for height scanner
- **Policy**: Used by critic for height scanner (in depth+scanner variant)

---

### 2. Off-Policy Reinforcement Learning (TD3/FastTD3)

#### Infrastructure
- **OffPolicyRunner** - New runner for off-policy algorithms in RSL-RL
- **TD3 Algorithm** - Twin Delayed Deep Deterministic Policy Gradient
- **FastTD3 Variant** - Optimized TD3 with faster critic updates

#### TD3 Configurations Added to Environments
- **Allegro Hand** - TD3 for dexterous manipulation
- **Ant** - TD3 and FastTD3 for quadruped locomotion
- **ANYmal-C** - TD3 for quadruped rough terrain (rough, flat, play variants)
- **Cartpole** - TD3 and FastTD3 for classic control
- **Franka Cabinet** - TD3 for manipulation tasks
- **Humanoid** - TD3 for bipedal locomotion
- **G1 Humanoid** - TD3 and FastTD3 for both rough and flat terrain
- **Spot Quadruped** - TD3 for Boston Dynamics Spot
- **Franka Lift** - TD3 for manipulation

#### Training Parameters
- **Random steps**: 5000 (exploration phase)
- **Gradient steps**: 20-40 (environment-dependent)
- **Replay buffer**: 1M transitions
- **Batch size**: 256
- **Learning rate**: 1e-4 (actor and critic)
- **Policy delay**: 2 (TD3), 1 (FastTD3)
- **Exploration noise**: 0.1

---

### 3. G1 Humanoid Vision-Based Locomotion

#### Environments
- **Isaac-Velocity-Rough-G1-Vision-v0** - Training with depth camera
- **Isaac-Velocity-Rough-G1-Vision-Play-v0** - Inference variant

#### Configuration
- **Depth Camera**: 53×30 resolution, 0.3-5.0m range
- **Attachment**: Pelvis link (torso)
- **Height Scanner**: 16×10 grid pattern (160 values)
- **Policy**: ActorCriticDepthCNN with separate CNN and MLP
- **Observations**:
  - Actor: proprio (123 dims) + depth (1590 dims) = 1713 dims
  - Critic: proprio + depth + height_scan = 1873 dims

#### G1 Specifications
- **Proprioception**: 123 dims (base velocities, gravity, commands, 37 joints × 2)
- **Hip Joint Penalty**: -0.2 weight (enhanced for vision control)
- **Agent Config**: `rsl_rl_depth_cnn_cfg.py`

#### G1 TD3 Support
- TD3 and FastTD3 available for rough and flat terrain
- Both standard PPO and off-policy TD3 options
- 4 total TD3 configs: G1RoughTD3, G1FlatTD3, G1RoughFastTD3, G1FlatFastTD3

---

### 4. H1 Humanoid Vision-Based Locomotion

#### Environments (8 total: 4 training + 4 play)

**Ground Truth Variants:**
- **Isaac-Velocity-Rough-H1-Vision-v0** - Depth CNN, ground truth state
- **Isaac-Velocity-Rough-H1-Vision-Play-v0** - Inference variant
- **Isaac-Velocity-Rough-H1-Vision-RNN-v0** - Depth CNN + LSTM, ground truth
- **Isaac-Velocity-Rough-H1-Vision-RNN-Play-v0** - Inference variant

**IMU Sensor Variants (Realistic State Estimation):**
- **Isaac-Velocity-Rough-H1-Vision-IMU-v0** - Depth CNN, IMU sensor
- **Isaac-Velocity-Rough-H1-Vision-IMU-Play-v0** - Inference variant
- **Isaac-Velocity-Rough-H1-Vision-IMU-RNN-v0** - Depth CNN + LSTM, IMU
- **Isaac-Velocity-Rough-H1-Vision-IMU-RNN-Play-v0** - Inference variant

#### Architecture Variants

| Variant | Vision | RNN | IMU | Parameters | GPU Memory | Use Case |
|---------|--------|-----|-----|------------|------------|----------|
| Standard PPO | No | No | No | ~403K | ~8 GB | Flat terrain, fast training |
| Depth CNN | Yes | No | No | ~1.84M | ~18 GB | Obstacle avoidance, reactive |
| Depth CNN + RNN | Yes | Yes | No | ~3.15M | ~24 GB | Complex obstacles, predictive |
| Depth CNN + IMU | Yes | No | Yes | ~1.84M | ~18 GB | Sim-to-real preparation |
| Depth CNN + RNN + IMU | Yes | Yes | Yes | ~3.15M | ~24 GB | Full stack, most realistic |

#### Sensors

**Depth Camera:**
- Attachment: torso_link
- Resolution: 53 × 30 pixels (1590 values)
- Range: 0.3 - 5.0m (extended from legged-loco's 2.0m)
- Pattern: Pinhole camera (focal_length=1.93, horizontal_aperture=3.8)
- Data type: distance_to_image_plane

**Height Scanner:**
- Attachment: torso_link
- Grid: 16 × 10 pattern (160 values)
- Coverage: 1.6m × 1.0m area
- Resolution: 0.1m grid spacing
- Data type: Ray-based height measurements

**IMU Sensor (Optional):**
- Attachment: torso_link
- Outputs: Angular velocity (3), Projected gravity (3)
- Update period: Every physics step (0.005s)
- Noise: Unoise(n_min=-0.1, n_max=0.1) for ang_vel, (-0.02, 0.02) for gravity
- Replaces: base_ang_vel and projected_gravity (linear velocity still ground truth)

#### Terrain Configuration (H1 Vision Variants)

Enhanced from legged-loco with increased obstacle density:

- **Pyramid stairs**: 15% (forward + 15% inverted)
- **Random boxes**: 35% (increased from 20%)
- **Random rough**: 5%
- **Slopes**: 5% (forward + 5% inverted)
- **Giant obstacles**: 40% (1.5m tall, 900 count - increased from 30)

**Total obstacle coverage:** ~75% (vs legged-loco's 40%)

#### Observations

**Ground Truth Variant:**
- Actor: proprio (69) + depth (1590) = 1659 dims
- Critic: proprio (69) + depth (1590) + policy/height_scan (256) = 1915 dims

**IMU Variant:**
- Same dimensions, different sources:
  - base_ang_vel → imu_ang_vel (from gyroscope)
  - projected_gravity → imu_gravity (from accelerometer)
  - base_lin_vel → kept as ground truth (simulates state estimator)

#### RNN Architecture (LSTM)

**Actor:**
```
Proprio (69) → MLP [512→256→128] ──┐
                                    ├→ Concat (256) → LSTM (256) → Action Head → Actions
Depth (1590) → CNN Backbone → 128 ─┘
```

**Critic:**
```
Proprio + Depth + Height (1915) → Encoder MLP [1915→512→256]
                                     ↓
                                   LSTM (256) → Critic MLP [256→512→256→128→1] → Value
```

**RNN Parameters:**
- Type: LSTM (or GRU)
- Hidden size: 256
- Layers: 1
- Maintains temporal context across timesteps

**Critic Encoder:**
- Maps raw observations (1915 dims) → RNN input size (256 dims)
- 2-layer MLP: 1915 → 512 → 256
- Required for dimension compatibility

---

## Modified Files

### Configuration Files

#### H1 Navigation Configurations
- **`source/isaaclab_tasks/.../navigation/config/h1/navigation_env_cfg.py`** (NEW)
  - H1NavigationEnvCfg - Base navigation (no vision)
  - H1NavigationDepthEnvCfg - Depth camera only
  - H1NavigationScannerEnvCfg - Height scanner only
  - H1NavigationDepthScannerEnvCfg - Depth + scanner combined
  - Separate observation groups for each sensor configuration
  - Hierarchical action configuration with pre-trained low-level policy

- **`source/isaaclab_tasks/.../navigation/config/h1/__init__.py`** (NEW)
  - Registered 12 navigation environments
  - Training and play variants for each sensor configuration
  - RNN variants for depth-based environments

- **`source/isaaclab_tasks/.../navigation/config/h1/agents/rsl_rl_ppo_cfg.py`** (NEW)
  - H1NavigationPPORunnerCfg - Base navigation without vision

- **`source/isaaclab_tasks/.../navigation/config/h1/agents/rsl_rl_depth_cfg.py`** (NEW)
  - H1NavigationDepthCNNRunnerCfg - Depth CNN configuration
  - Observation groups: actor ["proprio", "depth"], critic ["proprio", "depth"]

- **`source/isaaclab_tasks/.../navigation/config/h1/agents/rsl_rl_scanner_cfg.py`** (NEW)
  - H1NavigationScannerRunnerCfg - Height scanner MLP configuration
  - Observation groups: actor ["proprio", "scanner"], critic ["proprio", "scanner"]

- **`source/isaaclab_tasks/.../navigation/config/h1/agents/rsl_rl_depth_scanner_cfg.py`** (NEW)
  - H1NavigationDepthScannerRunnerCfg - Depth + scanner configuration
  - Observation groups: actor ["proprio", "depth"], critic ["proprio", "depth", "policy"]

- **`source/isaaclab_tasks/.../navigation/config/h1/agents/rsl_rl_depth_rnn_cfg.py`** (NEW)
  - H1NavigationDepthCNNRNNRunnerCfg - Depth CNN + LSTM configuration

- **`source/isaaclab_tasks/.../navigation/config/h1/agents/rsl_rl_depth_scanner_rnn_cfg.py`** (NEW)
  - H1NavigationDepthScannerRNNRunnerCfg - Depth + scanner + LSTM configuration

- **`source/isaaclab_tasks/.../navigation/config/__init__.py`**
  - Added import for H1 navigation configurations

#### H1 Locomotion Configurations
- **`source/isaaclab_tasks/.../config/h1/__init__.py`**
  - Registered 8 new vision-enabled environments
  - Added RNN and IMU environment categories

- **`source/isaaclab_tasks/.../config/h1/rough_vision_env_cfg.py`** (NEW)
  - H1RoughVisionEnvCfg - Base vision environment
  - H1RoughVisionEnvCfg_PLAY - Inference variant
  - H1VisionObservationsCfg - Observation specifications
  - H1VisionIMUObservationsCfg - IMU-based observations
  - H1RoughVisionIMUEnvCfg - IMU environment variant
  - H1RoughVisionIMUEnvCfg_PLAY - IMU inference variant

- **`source/isaaclab_tasks/.../config/h1/agents/rsl_rl_depth_cnn_cfg.py`** (NEW)
  - H1RoughDepthCNNRunnerCfg - PPO configuration for depth CNN
  - Observation groups: actor ["proprio", "depth"], critic ["proprio", "depth", "policy"]
  - Network: [512, 256, 128] hidden dims with ELU activation
  - num_actor_obs_prop: 69 (H1 proprioception)
  - obs_depth_shape: (53, 30)

- **`source/isaaclab_tasks/.../config/h1/agents/rsl_rl_depth_cnn_rnn_cfg.py`** (NEW)
  - H1RoughDepthCNNRecurrentRunnerCfg - PPO + RNN configuration
  - RNN parameters: LSTM, 256 hidden size, 1 layer
  - Critic encoder for observation dimensionality reduction
  - Same observation groups as depth CNN variant

#### G1 Configurations
- **`source/isaaclab_tasks/.../config/g1/__init__.py`**
  - Added TD3 and FastTD3 entry points to all environments
  - Registered 2 vision-enabled environments

- **`source/isaaclab_tasks/.../config/g1/rough_vision_env_cfg.py`** (NEW)
  - G1RoughVisionEnvCfg - Base vision environment
  - G1RoughVisionEnvCfg_PLAY - Inference variant
  - G1VisionObservationsCfg - Observation specifications

- **`source/isaaclab_tasks/.../config/g1/agents/rsl_rl_depth_cnn_cfg.py`** (NEW)
  - G1RoughDepthCNNRunnerCfg - PPO configuration for depth CNN
  - num_actor_obs_prop: 123 (G1 proprioception - 37 joints)
  - obs_depth_shape: (53, 30)

- **`source/isaaclab_tasks/.../config/g1/agents/rsl_rl_td3_cfg.py`** (NEW)
  - G1RoughTD3RunnerCfg - TD3 for rough terrain
  - G1FlatTD3RunnerCfg - TD3 for flat terrain
  - G1RoughFastTD3RunnerCfg - FastTD3 for rough terrain
  - G1FlatFastTD3RunnerCfg - FastTD3 for flat terrain

#### Other Environments (TD3 Support)
- Allegro Hand, Ant, ANYmal-C, Cartpole, Franka Cabinet, Humanoid, Spot, Franka Lift
- All received `rsl_rl_td3_cfg.py` and environment registration updates

### Infrastructure Files

- **`scripts/reinforcement_learning/rsl_rl/train.py`**
  - Added OffPolicyRunner support alongside OnPolicyRunner
  - Added directory existence check before checkpoint loading (fixes FileNotFoundError)

- **`scripts/reinforcement_learning/rsl_rl/play.py`**
  - Added OffPolicyRunner support for inference

- **`source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`** (NEW)
  - RslRlOffPolicyRunnerCfg - Base configuration for off-policy runners
  - RslRlTd3ActorCriticCfg - TD3 actor-critic configuration
  - RslRlTd3AlgorithmCfg - TD3 algorithm configuration
  - RslRlFastTd3AlgorithmCfg - FastTD3 algorithm configuration

- **`rsl_rl/rsl_rl/algorithms/td3.py`** (NEW)
  - TD3 algorithm implementation
  - FastTD3 variant implementation

- **`rsl_rl/rsl_rl/modules/td3_actor_critic.py`** (NEW)
  - ActorCriticTD3 - Actor-critic networks for TD3
  - Separate actor and twin critics (Q1, Q2)

---

## Fixed Issues

### RNN Integration Fixes

**1. Memory Initialization**
- **Issue**: `TypeError: Memory.__init__() got an unexpected keyword argument 'hidden_size'`
- **Fix**: Changed to positional arguments using `hidden_dim` parameter
- **File**: `rsl_rl/rsl_rl/modules/actor_critic_depth_cnn.py`

**2. Hidden State Attribute**
- **Issue**: `'Memory' object has no attribute 'hidden_states'`
- **Fix**: Changed all references from `hidden_states` to `hidden_state` (singular)
- **File**: `rsl_rl/rsl_rl/modules/actor_critic_depth_cnn.py`

**3. RNN Input Dimension Mismatch**
- **Issue**: `RuntimeError: input.size(-1) must be equal to input_size. Expected 256, got 1915`
- **Fix**: Added critic_encoder MLP (1915 → 512 → 256) before RNN
- **File**: `rsl_rl/rsl_rl/modules/actor_critic_depth_cnn.py`

**4. Critic MLP Dimension Mismatch**
- **Issue**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (4096x256 and 1915x512)`
- **Fix**: Recreated critic MLP with input size = rnn_hidden_size (256) instead of full obs (1915)
- **File**: `rsl_rl/rsl_rl/modules/actor_critic_depth_cnn.py`

**5. Method Parameter Names**
- **Issue**: `TypeError: got an unexpected keyword argument 'hidden_state'`
- **Fix**: Standardized parameter names to `hidden_state` (singular) across all methods
- **File**: `rsl_rl/rsl_rl/modules/actor_critic_depth_cnn.py`

### ONNX Export Fixes

**1. Depth CNN + RNN Export**
- **Issue**: `RuntimeError: shape '[-1, 53, 30]' is invalid for input of size 187`
- **Fix**:
  - Calculate full observation size for depth CNN (proprio + depth pixels)
  - Add encoding step in forward_lstm/forward_gru
  - Use action_head directly for depth CNN architectures
- **File**: `source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py`

### Training Script Fix

**1. Checkpoint Loading**
- **Issue**: `FileNotFoundError` when starting new experiments
- **Fix**: Added directory existence check before attempting to load checkpoints
- **File**: `scripts/reinforcement_learning/rsl_rl/train.py` (line 175)

---

## Documentation

### New Documentation Files

**1. `H1_NAVIGATION.md`** (NEW)
- Comprehensive navigation documentation for H1
- Hierarchical control architecture explanation
- All 12 environment variants with specifications
- Network architectures and parameter counts
- Training commands for all variants
- Sensor configuration details (depth camera, height scanner)
- Observation group explanations
- GPU memory requirements
- Comparison: locomotion vs navigation tasks
- Prerequisites (low-level policy training)
- Architecture selection guide

**2. `H1_ARCHITECTURES.md`**
- Comprehensive architecture documentation for H1 variants
- Architecture overview and comparison table
- Detailed specifications for all 3 variants (Standard PPO, Depth CNN, Depth CNN + RNN)
- Parameter breakdown: 403K → 1.84M → 3.15M parameters
- Scaling analysis and memory requirements
- Training commands and architecture selection guide
- Observation group explanations
- Terrain configuration details
- Performance comparison metrics
- File locations reference

**3. `CHANGELOG_H1.md`**
- H1-specific changelog
- Complete list of H1 vision environments
- Configuration file changes
- Bug fixes and infrastructure improvements

**4. `CHANGELOG.md`** (this file)
- Comprehensive changelog covering all additions
- TD3/FastTD3 support across all environments
- G1 and H1 vision configurations
- Infrastructure and fix documentation

---

## Comparison with Legged-Loco

### Inspired by Legged-Loco
- Depth camera resolution (53×30)
- Height scanner configuration
- Terrain generation approach
- PPO hyperparameters
- Depth CNN architecture concept

### Enhanced/Different
- **Novel hybrids**:
  - H1 robot + G1 vision (legged-loco H1 vision uses only height scanner)
  - G1 robot + extended depth range
- **Extended range**: 5.0m vs 2.0m depth camera range
- **RNN support**: Temporal memory (legged-loco H1 vision is reactive only)
- **IMU variant**: Realistic sensor option for sim-to-real
- **Higher difficulty**: 75% vs 40% obstacle coverage (H1)
- **Critic encoder**: Custom encoder for RNN variant
- **Toggle options**: Ground truth vs IMU selectable
- **TD3/FastTD3**: Off-policy learning not in legged-loco

### Novel Contributions
- Combined H1 morphology + G1 vision + extended range + RNN + IMU
- IMU/ground-truth toggle for sim-to-real preparation
- RNN integration with depth CNN for humanoid locomotion
- Off-policy (TD3/FastTD3) support across multiple environments
- Production-ready configuration for advanced research

---

## Breaking Changes

None - All additions are new environments and configurations. Existing environments unchanged.

---

## Training Commands

### H1 Navigation Variants

**Base Navigation (No Vision):**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-v0 \
  --headless
```

**Depth Camera Only:**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-Depth-v0 \
  --headless
```

**Depth Camera + RNN:**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-Depth-RNN-v0 \
  --headless
```

**Height Scanner Only:**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-Scanner-v0 \
  --headless
```

**Depth + Scanner:**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-DepthScanner-v0 \
  --headless
```

**Depth + Scanner + RNN (Full Stack):**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Navigation-Flat-H1-DepthScanner-RNN-v0 \
  --headless
```

### H1 Locomotion Vision Variants

**Standard PPO:**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-v0 \
  --headless
```

**Depth CNN (Ground Truth):**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-Vision-v0 \
  --headless
```

**Depth CNN + RNN (Ground Truth):**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-Vision-RNN-v0 \
  --headless
```

**Depth CNN (IMU):**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-Vision-IMU-v0 \
  --headless
```

**Depth CNN + RNN (IMU):**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-H1-Vision-IMU-RNN-v0 \
  --headless
```

### G1 Vision Variants

**Depth CNN (PPO):**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-Vision-v0 \
  --headless
```

**TD3:**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-v0 \
  --agent rsl_rl_td3_cfg \
  --headless
```

**FastTD3:**
```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-v0 \
  --agent rsl_rl_fast_td3_cfg \
  --headless
```

### GPU Usage Reduction
```bash
# Add these flags to any training command
--num_envs 1024          # Reduce parallel environments
--headless               # Disable GUI
```

---

## Environment Registry Summary

### H1 Humanoid Locomotion (12 total)

**Standard (4 existing):**
1. `Isaac-Velocity-Rough-H1-v0`
2. `Isaac-Velocity-Rough-H1-Play-v0`
3. `Isaac-Velocity-Flat-H1-v0`
4. `Isaac-Velocity-Flat-H1-Play-v0`

**Vision - Ground Truth (4 new):**
5. `Isaac-Velocity-Rough-H1-Vision-v0`
6. `Isaac-Velocity-Rough-H1-Vision-Play-v0`
7. `Isaac-Velocity-Rough-H1-Vision-RNN-v0`
8. `Isaac-Velocity-Rough-H1-Vision-RNN-Play-v0`

**Vision - IMU (4 new):**
9. `Isaac-Velocity-Rough-H1-Vision-IMU-v0`
10. `Isaac-Velocity-Rough-H1-Vision-IMU-Play-v0`
11. `Isaac-Velocity-Rough-H1-Vision-IMU-RNN-v0`
12. `Isaac-Velocity-Rough-H1-Vision-IMU-RNN-Play-v0`

### H1 Humanoid Navigation (12 total)

**Base Navigation (2)**:
1. `Isaac-Navigation-Flat-H1-v0`
2. `Isaac-Navigation-Flat-H1-Play-v0`

**Depth Camera Only (4)**:
3. `Isaac-Navigation-Flat-H1-Depth-v0`
4. `Isaac-Navigation-Flat-H1-Depth-Play-v0`
5. `Isaac-Navigation-Flat-H1-Depth-RNN-v0`
6. `Isaac-Navigation-Flat-H1-Depth-RNN-Play-v0`

**Height Scanner Only (2)**:
7. `Isaac-Navigation-Flat-H1-Scanner-v0`
8. `Isaac-Navigation-Flat-H1-Scanner-Play-v0`

**Depth + Scanner Combined (4)**:
9. `Isaac-Navigation-Flat-H1-DepthScanner-v0`
10. `Isaac-Navigation-Flat-H1-DepthScanner-Play-v0`
11. `Isaac-Navigation-Flat-H1-DepthScanner-RNN-v0`
12. `Isaac-Navigation-Flat-H1-DepthScanner-RNN-Play-v0`

### G1 Humanoid Locomotion (6 total)

**Standard (4 existing):**
1. `Isaac-Velocity-Rough-G1-v0` (PPO, TD3, FastTD3)
2. `Isaac-Velocity-Rough-G1-Play-v0`
3. `Isaac-Velocity-Flat-G1-v0` (PPO, TD3, FastTD3)
4. `Isaac-Velocity-Flat-G1-Play-v0`

**Vision (2 new):**
5. `Isaac-Velocity-Rough-G1-Vision-v0`
6. `Isaac-Velocity-Rough-G1-Vision-Play-v0`

### Other Environments with TD3 Support

- Allegro Hand (TD3)
- Ant (TD3, FastTD3)
- ANYmal-C (TD3 - rough, flat, play)
- Cartpole (TD3, FastTD3)
- Franka Cabinet (TD3)
- Humanoid (TD3)
- Spot (TD3)
- Franka Lift (TD3)

---

## Known Issues

None currently.

---

## Future Work

### H1 Navigation
- [ ] IMU sensor variants
- [ ] Rough terrain navigation (requires rough low-level policy)
- [ ] Dynamic obstacle avoidance
- [ ] Multi-goal planning
- [ ] TD3/FastTD3 support for off-policy training
- [ ] Multi-camera configurations
- [ ] Semantic segmentation integration

### H1 Locomotion
- [ ] Add GRU variant (currently only LSTM)
- [ ] Velocity estimation from IMU acceleration (currently uses ground truth)
- [ ] Multi-camera configurations
- [ ] Attention mechanisms for vision processing
- [ ] Distillation from privileged teacher
- [ ] TD3/FastTD3 support

### G1 Locomotion
- [ ] RNN/LSTM support for vision variant
- [ ] IMU sensor variant
- [ ] Multi-camera configurations

### General
- [ ] Vision support for more robots
- [ ] Unified vision configuration system
- [ ] Performance benchmarking suite

---

**Contributors:** Implementation based on legged-loco foundation with significant enhancements
**Date:** January 2025
**IsaacLab Version:** 1.0.0+
**License:** BSD-3-Clause
