# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Agilix Ranger Mini omni-wheel mobile robot.

The following configurations are available:

* :obj:`RANGER_MINI_CFG`: Agilix Ranger Mini omni-wheel robot

Reference: https://www.agilex.ai/products/ranger-mini
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Get the directory where this file is located
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

##
# Configuration
##

RANGER_MINI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_CURRENT_DIR}/ranger_mini.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),
        joint_pos={
            # 4 steering joints (revolute, position control)
            "fl_steering_wheel": 0.0,
            "fr_steering_wheel": 0.0,
            "rl_steering_wheel": 0.0,
            "rr_steering_wheel": 0.0,
            # 4 drive joints (continuous, velocity control)
            "fl_wheel": 0.0,
            "fr_wheel": 0.0,
            "rl_wheel": 0.0,
            "rr_wheel": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=[".*_steering_wheel"],
            effort_limit=5.0,        # From URDF
            velocity_limit=6.28,     # From URDF
            stiffness=0.0,
            damping=5000.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel"],
            effort_limit=100.0,      # From URDF
            velocity_limit=10.0,     # From URDF
            stiffness=0.0,
            damping=10000.0,  # High damping for ground constraint
        ),
    },
)
"""Configuration of Agilix Ranger Mini omni-wheel robot using implicit actuator models.

The Ranger Mini uses 4 independent omni-directional wheels, each with:
- 1 steering joint (for wheel orientation): fl_steering_wheel, fr_steering_wheel, rl_steering_wheel, rr_steering_wheel
- 1 drive joint (for wheel rotation): fl_wheel, fr_wheel, rl_wheel, rr_wheel
Total: 8 joints

This configuration uses position control for steering joints and velocity control for drive joints.
"""
