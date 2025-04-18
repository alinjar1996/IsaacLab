# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a husky. It combines the concepts of
scene, action, observation and event managers to create an environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_husky_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import numpy
from scipy.spatial.transform import Rotation as R

import argparse

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch
import time

import pandas as pd


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg

from isaaclab_tasks.manager_based.classic.husky.husky_env_cfg import HuskySceneCfg

# Pre-defined configs
##
from isaaclab_assets.robots.husky import HUSKY_CFG  # isort:skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip




##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*_wheel"], scale=5.0)

    joint_positions = mdp.JointPositionActionCfg(asset_name="robot",joint_names=[".*_wheel"],scale=1.0)



@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
         # Add contact forces observation
        # contact_forces = ObsTerm(func=mdp.contact_forces,
        #      params={
        #     "threshold": 0.01,  # Minimum force threshold to register a contact
        #     "sensor_cfg": SceneEntityCfg("robot", body_names=["front_left_wheel_link", "front_right_wheel_link", 
        #                                                        "rear_left_wheel_link", "rear_right_wheel_link"])
        # })

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # # on startup
    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #         "mass_distribution_params": (0.1, 0.5),
    #         "operation": "add",
    #     },
    # )

    # # on reset
    # reset_wheel_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel"]),
    #         "position_range": (-1.0, 1.0),
    #         "velocity_range": (-0.1, 0.1),
    #     },
    # )




@configclass
class HuskyEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = HuskySceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # # simulation settings
        self.sim.dt = 0.05  # sim step every 5ms: 200Hz
        self.enable_corruption = True
        self.concatenate_terms = True


import pandas as pd
import torch

# Global variable to control whether to write the header
header_written = False
header_written_pos = False
header_written_base = False

import pandas as pd
import torch



def main():
    """Main function."""
    global header_written
    global header_written_pos
    global header_written_base
    # parse the arguments
    env_cfg = HuskyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)
    force_sensor = True
    position_sensor = True
    imu_sensor = True

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 30000 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            # Sample random actions
            joint_efforts = torch.full_like(env.action_manager.action, -10)
            joint_efforts[:, [0, 2]] = -10.0  # left wheels
            joint_efforts[:, [1, 3]] = 10.0   # right wheels

            # Inside while loop in main()
            target_position = torch.full_like(env.action_manager.action, count * 0.0)  # slowly increases over time
            obs, _ = env.step(target_position)


            # Print contact forces every 1 step
            if count % 1 == 0:
                try:
                    # Access contact forces using dictionary-style access
                    print("-" * 40)

                    if force_sensor == True:
                        # Access the contact force data
                        contact_forces = env.scene["contact_forces"].data.net_forces_w



                        # Move tensor to CPU before converting to NumPy
                        contact_forces_cpu = contact_forces.cpu().numpy()

                        

                        # Reshape the tensor from (1, 4, 3) to (4, 3)
                        contact_forces_reshaped = contact_forces_cpu[0]  # The first (and only) element contains the forces for 4 wheels
                        

                        # Create a dictionary to organize the forces by wheel
                        data_dict = {
                            'Count': count,
                            'fl_wheel_Force_x': contact_forces_reshaped[0, 0],
                            'fl_wheel_Force_y': contact_forces_reshaped[0, 1],
                            'fl_wheel_Force_z': contact_forces_reshaped[0, 2],
                            'fr_wheel_Force_x': contact_forces_reshaped[1, 0],
                            'fr_wheel_Force_y': contact_forces_reshaped[1, 1],
                            'fr_wheel_Force_z': contact_forces_reshaped[1, 2],
                            'bl_wheel_Force_x': contact_forces_reshaped[2, 0],
                            'bl_wheel_Force_y': contact_forces_reshaped[2, 1],
                            'bl_wheel_Force_z': contact_forces_reshaped[2, 2],
                            'br_wheel_Force_x': contact_forces_reshaped[3, 0],
                            'br_wheel_Force_y': contact_forces_reshaped[3, 1],
                            'br_wheel_Force_z': contact_forces_reshaped[3, 2],
                        
                        }

                        # Convert the dictionary to a DataFrame
                        contact_forces_df = pd.DataFrame([data_dict])
                        

                        # Open the CSV file in append mode, write header only once
                        csv_filename = 'contact_forces.csv'
                        with open(csv_filename, mode='a', newline='') as file:
                            if not header_written:
                                contact_forces_df.to_csv(file, header=True, index=False)
                                header_written = True
                            else:
                                contact_forces_df.to_csv(file, header=False, index=False)

                        print(f"Contact forces saved to {csv_filename}")              

                        print("Contact forces shape:", contact_forces.shape)

                        # Print forces for each wheel
                        print("FL_wheel Contact Forces - X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(contact_forces_reshaped[0, 0], contact_forces_reshaped[0, 1], contact_forces_reshaped[0, 2]))
                        print("FR_wheel Contact Forces - X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(contact_forces_reshaped[1, 0], contact_forces_reshaped[1, 1], contact_forces_reshaped[1, 2]))
                        print("BL_wheel Contact Forces - X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(contact_forces_reshaped[2, 0], contact_forces_reshaped[2, 1], contact_forces_reshaped[2, 2]))
                        print("BR_wheel Contact Forces - X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(contact_forces_reshaped[3, 0], contact_forces_reshaped[3, 1], contact_forces_reshaped[3, 2]))
                        
                        # Calculate sum of forces in X, Y, and Z directions across all wheels
                        x_sum = contact_forces_reshaped[:, 0].sum()
                        y_sum = contact_forces_reshaped[:, 1].sum()
                        z_sum = contact_forces_reshaped[:, 2].sum()

                        # Print the summed forces
                        print("Total Forces - X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(x_sum, y_sum, z_sum))

                        print("Contact forces:", contact_forces.tolist())  # Convert tensor to list for clean output
                        print("Max contact force:", torch.max(contact_forces).item())
                        print("-" * 40)

                    if position_sensor == True:
                        ## Pos
                        contact_positions = env.scene["contact_forces"].data.pos_w

                        print("Type:", type(contact_positions))

                        contact_positions_cpu = contact_positions.cpu().numpy()

                        contact_positions_reshaped = contact_positions_cpu[0]


                        data_dict_pos ={
                            'Count': count,
                            'fl_wheel_Position_x': contact_positions_reshaped[0, 0],
                            'fl_wheel_Position_y': contact_positions_reshaped[0, 1],
                            'fl_wheel_Position_z': contact_positions_reshaped[0, 2],
                            'fr_wheel_Position_x': contact_positions_reshaped[1, 0],
                            'fr_wheel_Position_y': contact_positions_reshaped[1, 1],
                            'fr_wheel_Position_z': contact_positions_reshaped[1, 2],
                            'bl_wheel_Position_x': contact_positions_reshaped[2, 0],
                            'bl_wheel_Position_y': contact_positions_reshaped[2, 1],
                            'bl_wheel_Position_z': contact_positions_reshaped[2, 2],
                            'br_wheel_Position_x': contact_positions_reshaped[3, 0],
                            'br_wheel_Position_y': contact_positions_reshaped[3, 1],
                            'br_wheel_Position_z': contact_positions_reshaped[3, 2],
                        }
                        
                        contact_pos_df = pd.DataFrame([data_dict_pos])

                        csv_filename_pos = 'contact_positions.csv'
                        with open(csv_filename_pos, mode='a', newline='') as file:
                            if not header_written_pos:
                                contact_pos_df.to_csv(file, header=True, index=False)
                                header_written_pos = True
                            else:
                                contact_pos_df.to_csv(file, header=False, index=False)  

                        print(f"Contact positions saved to {csv_filename_pos}")

                        #print("Contact positions shape:", contact_positions.shape())

                        print("Contact positions:", contact_positions.tolist())  # Convert tensor to list for clean output

                        # Print positions for each wheel
                        print("FL_wheel Contact positions - X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(contact_positions_reshaped[0, 0], contact_positions_reshaped[0, 1], contact_positions_reshaped[0, 2]))
                        print("FR_wheel Contact positions - X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(contact_positions_reshaped[1, 0], contact_positions_reshaped[1, 1], contact_positions_reshaped[1, 2]))
                        print("BL_wheel Contact positions - X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(contact_positions_reshaped[2, 0], contact_positions_reshaped[2, 1], contact_positions_reshaped[2, 2]))
                        print("BR_wheel Contact positions - X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(contact_positions_reshaped[3, 0], contact_positions_reshaped[3, 1], contact_positions_reshaped[3, 2]))
                        print("-" * 40)
                    if imu_sensor == True:
                        
                        # Fetch the robot position and quaternion from the environment
                        robot_pos = env.scene["imu_data"].data.pos_w
                        robot_quat = env.scene["imu_data"].data.quat_w  # quaternion in [w, x, y, z]

                        # Move the tensors to CPU before converting to NumPy arrays
                        robot_pos_cpu = robot_pos.cpu().numpy()  # Move to CPU and convert to numpy
                        robot_quat_cpu = robot_quat.cpu().numpy()  # Move to CPU and convert to numpy

                        print("robot_pos_cpu", robot_pos_cpu)
                        print("robot_quat_cpu", robot_quat_cpu)

                        # Convert [w, x, y, z] to [x, y, z, w] format
                        robot_quat_xyzw = numpy.array([robot_quat_cpu[0][1], robot_quat_cpu[0][2], robot_quat_cpu[0][3], robot_quat_cpu[0][0]])
                        print("robot_quat_xyzw", robot_quat_xyzw)

                        # Convert quaternion to roll-pitch-yaw (RPY) in radians
                        robot_rpy = R.from_quat(robot_quat_xyzw).as_euler('xyz', degrees=False)

                        # Reshape robot position (if needed)
                        robot_pos_reshaped = robot_pos_cpu[0]

                        # robot_rpy_cpu = robot_rpy (no need to move to CPU again, already in numpy format)
                        robot_rpy_cpu = robot_rpy
                        robot_rpy_reshaped = robot_rpy_cpu

                        print("robot_rpy_reshaped", robot_rpy_reshaped)

                        # Prepare data for saving to CSV
                        data_dict_base = {
                            'Count': count,  # Replace with actual count if necessary
                            'Robot_Position_x': robot_pos_reshaped[0],
                            'Robot_Position_y': robot_pos_reshaped[1],
                            'Robot_Position_z': robot_pos_reshaped[2],
                            'Robot_Orientation_roll': robot_rpy_reshaped[0],
                            'Robot_Orientation_pitch': robot_rpy_reshaped[1],
                            'Robot_Orientation_yaw': robot_rpy_reshaped[2],
                        }

                        # Create DataFrame
                        base_pose_df = pd.DataFrame([data_dict_base])

                        # Save to CSV
                        csv_filename_base = 'base_pose.csv'
                        with open(csv_filename_base, mode='a', newline='') as file:
                            if not header_written_base:
                                base_pose_df.to_csv(file, header=True, index=False)
                                header_written_base = True
                            else:
                                base_pose_df.to_csv(file, header=False, index=False)

                        print(f"Base Pose saved to {csv_filename_base}")

                except Exception as e:
                    print(f"Error accessing data: {e}")

            # Increment counter
            count += 1

    # Close the environment
    env.close()



if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation app
    simulation_app.close()
