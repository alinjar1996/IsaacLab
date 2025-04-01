# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with a four-wheeled robot.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene_husky.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using a four-wheeled robot with the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--urdf_path", type=str, default="/home/ims/isaacsim/IsaacLab/husky_description/urdf/husky_temp.urdf", help="Path to your four-wheeled robot URDF file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


@configclass
class FourWheeledRobotCfg(ArticulationCfg):
    """Configuration for a four-wheeled robot loaded from URDF."""

    # If you have a USD file for your robot, use this
    spawn = sim_utils.UsdFileCfg(
        usd_path=args_cli.urdf_path if args_cli.urdf_path else "/home/ims/isaacsim/IsaacLab/husky_description/urdf/husky_temp.usd",
    )

    # Optional: Define other properties for your robot
    init_state = {"joint_pos": {}, "joint_vel": {}}


@configclass
class FourWheeledRobotSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with a four-wheeled robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Define our four-wheeled robot
    robot = FourWheeledRobotCfg(prim_path="{ENV_REGEX_NS}/Robot")




def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]
    
    # Find wheel joint names
    # This assumes your URDF has wheel joints that contain "wheel" in their name
    wheel_joints = [joint for joint in robot.actuated_joint_names if "wheel" in joint.lower()]
    
    if not wheel_joints:
        print("Warning: No wheel joints found. Please check your URDF or joint naming.")
        # Fallback to all joints
        wheel_joints = robot.actuated_joint_names
    
    print(f"Controlling these joints: {wheel_joints}")
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            root_state = torch.zeros((scene.num_envs, 13), device=scene.device)
            # Position: x, y, z (slightly above ground to prevent collision issues)
            root_state[:, :3] = torch.tensor([0.0, 0.0, 0.1], device=scene.device)
            # Rotation: quaternion (w, x, y, z) - default orientation
            root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=scene.device)
            # Add environment origins offset
            root_state[:, :3] += scene.env_origins
            
            # Apply root state
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # Reset joint positions to defaults from URDF
            # If your robot has explicit defaults, use those
            joint_pos = torch.zeros_like(robot.data.joint_pos)
            joint_vel = torch.zeros_like(robot.data.joint_vel)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        
        # Apply actions to the robot wheels
        # Create simple motion pattern: forward motion
        if count % 200 < 100:
            # Move forward
            wheel_velocities = torch.ones_like(robot.data.joint_vel) * 5.0
        else:
            # Turn
            # For differential drive, set opposite velocities for left and right wheels
            wheel_velocities = torch.ones_like(robot.data.joint_vel) * 2.0
            # Identify left vs right wheels (assuming naming convention with "left" or "right" in name)
            for i, name in enumerate(robot.actuated_joint_names):
                if "left" in name.lower():
                    wheel_velocities[:, i] *= 1.0
                elif "right" in name.lower():
                    wheel_velocities[:, i] *= -1.0
        
        # Set velocity targets for the joints
        robot.set_joint_velocity_target(wheel_velocities)
        
        # Write data to sim
        scene.write_data_to_sim()
        
        # Perform step
        sim.step()
        
        # Increment counter
        count += 1
        
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view([5.0, 0.0, 3.0], [0.0, 0.0, 0.5])
    
    # Design scene
    scene_cfg = FourWheeledRobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print(f"[INFO]: Loaded robot URDF from: {args_cli.urdf_path}")
    
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()