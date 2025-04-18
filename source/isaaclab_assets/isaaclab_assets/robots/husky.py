# Copyright (c) 2025, Alinjar Dan, Institute of Technology, university of Tartu 
# All rights reserved.
#


"""Configuration for the wheeled robot husky"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg, actuator_pd
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Configuration
##

HUSKY_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/husky",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/ims/isaacsim/IsaacLab/husky_description/urdf/husky_temp/husky_temp.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
    ),
    actuators={
        "front_left_wheel": ActuatorBaseCfg(
            joint_names_expr="front_left_wheel",
            effort_limit=100.0,
            class_type=actuator_pd.ImplicitActuator,
            #class_type=actuator_pd.IdealPDActuator,
            stiffness=500.0,
            damping=50.0
        ),
        "front_right_wheel": ActuatorBaseCfg(
            joint_names_expr="front_right_wheel",
            effort_limit=100.0,
            class_type=actuator_pd.ImplicitActuator,
            stiffness=500.0,
            damping=50.0
        ),
        "rear_left_wheel": ActuatorBaseCfg(
            joint_names_expr="rear_left_wheel",
            effort_limit=100.0,
            class_type=actuator_pd.ImplicitActuator,
            stiffness=500.0,
            damping=50.0
        ),
        "rear_right_wheel": ActuatorBaseCfg(
            joint_names_expr="rear_right_wheel",
            effort_limit=100.0,
            class_type=actuator_pd.ImplicitActuator,
            stiffness=500.0,
            damping=50.0
        )
      }
)
"""Configuration for the husky."""
