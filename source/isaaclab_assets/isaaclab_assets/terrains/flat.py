# Copyright (c) 2025, Alinjar Dan, Institute of Technology, university of Tartu 
# All rights reserved.
#


"""Configuration for the wheeled robot husky"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg, actuator_pd
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Configuration
##
# terrain = AssetBaseCfg(
#         prim_path="/World/ground",
#         spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
#     )
FLAT_TERRAIN_CFG = AssetBaseCfg(
    prim_path="/World/terrain_flat",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/ims/isaacsim/IsaacLab/terrain_description/terrain_flat.usd",
       )
)
