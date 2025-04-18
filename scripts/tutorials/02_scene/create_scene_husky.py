# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with Husky.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_husky_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher



# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface with Husky.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.actuators import actuator_pd

@configclass
class HuskySceneCfg(InteractiveSceneCfg):
    """Configuration for a Husky scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # # Husky asset
    # husky = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/Husky",
    #     spawn=sim_utils.UsdFileCfg(usd_path="/home/ims/isaacsim/IsaacLab/husky_description/husky_temp.usd")
    # )
    
    # Husky asset
    husky = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path="/home/ims/isaacsim/IsaacLab/husky_description/urdf/husky_temp/husky_temp.usd"),
    actuators={
        "front_left_wheel": ActuatorBaseCfg(
            joint_names_expr="front_left_wheel",
            effort_limit=100.0,
            class_type=actuator_pd.IdealPDActuator,
            #class_type=actuator_pd.IdealPDActuator,
            stiffness=500.0,
            damping=50.0
        ),
        "front_right_wheel": ActuatorBaseCfg(
            joint_names_expr="front_right_wheel",
            effort_limit=100.0,
            class_type=actuator_pd.IdealPDActuator,
            stiffness=500.0,
            damping=50.0
        ),
        "rear_left_wheel": ActuatorBaseCfg(
            joint_names_expr="rear_left_wheel",
            effort_limit=100.0,
            class_type=actuator_pd.IdealPDActuator,
            stiffness=500.0,
            damping=50.0
        ),
        "rear_right_wheel": ActuatorBaseCfg(
            joint_names_expr="rear_right_wheel",
            effort_limit=100.0,
            class_type=actuator_pd.IdealPDActuator,
            stiffness=500.0,
            damping=50.0
        )
      }
    )



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    husky = scene["husky"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = husky.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            husky.write_root_pose_to_sim(root_state[:, :7])
            husky.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = husky.data.default_joint_pos.clone(), husky.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            husky.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting Husky state...")
        efforts = torch.randn_like(husky.data.joint_pos) * 5.0
        husky.set_joint_effort_target(efforts)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_cfg = HuskySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
