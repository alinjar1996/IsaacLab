# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg


import isaaclab_tasks.manager_based.classic.husky.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.husky import HUSKY_CFG  # isort:skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.terrains.config.flat import FLAT_TERRAIN_CFG
from isaaclab.terrains.config.slope import SLOPE_TERRAIN_CFG
from pxr import Usd, UsdGeom
import omni.usd
##
# Scene definition
##


@configclass
class HuskySceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    #ground plane terrain
    # terrain = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    # )

    terrain = AssetBaseCfg(prim_path="/World/ground",
            spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/ims/isaacsim/IsaacLab/terrain_description/cube_4.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # example quaternion
            ),
            collision_group=-1,
            debug_vis=True,
        )
#     terrain = AssetBaseCfg(
#     prim_path="/World/terrain_flat",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/ims/isaacsim/IsaacLab/terrain_description/terrain_flat.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=True,  # ✅ prevents falling
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_depenetration_velocity=10.0
#         ),
#         collision_props=sim_utils.CollisionPropertiesCfg(
#             collision_enabled=True,
#             contact_offset=0.02,
#             rest_offset=0.0
#         )
#     ),
#     debug_vis=True,
# )
    # ## add terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=FLAT_TERRAIN_CFG, #ROUGH_TERRAINS_CFG
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=0.2,
    #         dynamic_friction=0.2,
    #     ),
    #     debug_vis=False,
    # )


    # Husky
    robot: ArticulationCfg = HUSKY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    
    #sensors

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_wheel_link", 
        update_period=0.01, 
        history_length=6, 
        debug_vis=False,
        track_pose=True
    )

    imu_data = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.01,
        history_length=6,
        debug_vis=True,
        gravity_bias=(0, 0, 9.81)
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*_wheel"], scale=100.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_wheel_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel"]),
            "position_range": (-3.0, 3.0),
            "velocity_range": (-2.5, 2.5),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["front_left_wheel"]), "target": 0.0},
    # )
    # # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["front_left_wheel"])},
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["front_left_wheel"])},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel"]), "bounds": (-3.0, 3.0)},
    )


##
# Environment configuration
##


@configclass
class HuskyEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: HuskySceneCfg = HuskySceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
