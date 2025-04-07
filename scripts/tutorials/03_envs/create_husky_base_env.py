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


import argparse

from isaaclab.app import AppLauncher
#from isaaclab.sensors import ContactSensorCfg

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


def main():
    """Main function."""
    # parse the arguments
    env_cfg = HuskyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            #joint_efforts = torch.randn_like(env.action_manager.action)
            joint_efforts = torch.full_like(env.action_manager.action, -10)
            joint_efforts[:, [0, 2]] = -10.0  # left wheels
            joint_efforts[:, [1, 3]] = 10.0   # right wheels
            #joint_efforts = torch.zeros_like(env.action_manager.action)
            #print("joint_efforts", joint_efforts)
            # step the environment
            #start = time.time()
            # Inside while loop in main()
            target_position = torch.full_like(env.action_manager.action, count * 0.025)  # slowly increases over time
            obs, _ = env.step(target_position)
            #obs, _ = env.step(joint_efforts)
            #end = time.time()
            #time_taken = end - start
            #print("time_taken", time_taken)
            #print("count", count)
            # print current orientation of pole
            #print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
           #print("[Env 1]: Pole joint: ", obs["policy"][1][1].item())
            # update counter
            # Print contact forces every 20 steps
            if count % 1 == 0:
                try:
                    # Access contact forces using dictionary-style access
                    print("-" * 40)
                    #print("[INFO] Contact forces at step", count)
                    #print("Contact sensor info:", env.scene["contact_forces"])
                    # Access the contact force data
                    contact_forces = env.scene["contact_forces"].data.net_forces_w
                    print("Contact forces shape:", contact_forces.shape)
                    print("Contact forces:", contact_forces.tolist())  # Convert tensor to list for clean output
                    print("Max contact force:", torch.max(contact_forces).item())
                    print("-" * 40)
                    # Calculate force sums for all robots
                    num_envs = contact_forces.shape[0]  # Get number of environments from tensor shape
                    print(f"Processing contact forces for {num_envs} robots:")

                    for robot_idx in range(num_envs):
                        # Sum X, Y, Z components separately for each robot
                        x_sum = contact_forces[robot_idx, :, 0].sum().item()
                        y_sum = contact_forces[robot_idx, :, 1].sum().item()
                        z_sum = contact_forces[robot_idx, :, 2].sum().item()
                        
                        print(f"Robot {robot_idx} total forces - X: {x_sum:.2f}, Y: {y_sum:.2f}, Z: {z_sum:.2f}")
                except Exception as e:
                    print(f"Error accessing contact forces: {e}")
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
