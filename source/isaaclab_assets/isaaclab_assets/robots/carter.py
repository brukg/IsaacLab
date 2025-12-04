# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the NVIDIA Carter differential drive mobile robot.

The following configurations are available:

* :obj:`CARTER_V1_CFG`: NVIDIA Carter V1 differential drive robot

Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/robots/carter.html
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

CARTER_V1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Carter/carter_v1.usd",
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
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            "left_wheel": 0.0,
            "right_wheel": 0.0,
            "rear_pivot": 0.0,
            "rear_axle": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel", "right_wheel"],
            effort_limit=200.0,         # Increased for terrain climbing
            velocity_limit=20.0,        # Reasonable wheel speed limit
            stiffness=0.0,              # OK for velocity control
            damping=10000.0,            # High damping for ground constraint (like Ridgeback)
        ),
    },
)
"""Configuration of NVIDIA Carter V1 differential drive robot using implicit actuator models.

The Carter robot uses differential drive with two wheels for locomotion.
This configuration uses velocity control for the wheel joints.
"""
