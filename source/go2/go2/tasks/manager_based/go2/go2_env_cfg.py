# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.actuators import IdealPDActuatorCfg  




from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

##
# Scene definition
##


def base_quat(obs_manager, scene, env_ids):
    return scene["robot"].data.root_quat_wxyz[env_ids]


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "legs_pd": IdealPDActuatorCfg(
                joint_names_expr=[
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                ],
                stiffness=30,   # Kp (paper)
                damping=1,      # Kd (paper)
            )
        }
        
        )

    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    


    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##
BAD_CONTACTS = SceneEntityCfg("contact_forces",
    body_names="(base|FL_thigh|FR_thigh|FL_calf|FR_calf)", 
    )
ALLOWED_EARLY = SceneEntityCfg(
    "contact_forces",
    body_names="(RL_foot|RR_foot|FL_foot|FR_foot)",   
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ],
        scale=0.15, 
    )

#     joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names= [
#     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
#     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
#     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
#     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
# ]
# , scale=20.0
#     )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale = 2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale = 0.25)
        projected_gravity = ObsTerm(func = mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, scale = 1.0)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale = 0.05)
        previous = ObsTerm(func = mdp.last_action)
        previous_previous = ObsTerm(func = mdp.prev_action)

        #base_orientation = ObsTerm(func=base_quat)
#         base_orientation = ObsTerm(
#         func=base_quat,
#         params={"scene": "scene", "env_ids": "env_ids"},
# )



        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
#     reset_robot_joints = EventTerm(
#         func=mdp.reset_joints_by_offset,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=[
#         "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
#         "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
#         "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
#         "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
# ]),
#         "position_range": 
#         (-0.1, -0.1),
#         "velocity_range": (-0.1 * math.pi, 0.1 * math.pi),
#         },
#     )
    # reset_robot_joints = EventTerm(
    #         func=mdp.reset_joints_by_offset,
    #         mode="reset",
    #         params={
    #             "asset_cfg": SceneEntityCfg("robot", joint_names=[
    #         "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    #         "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    #         "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    #         "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    # ]),
    #         "position_range": 
    #         (-0, 0),
    #         "velocity_range": (-0, 0),
    #         },
    #     )

    reset_robot_joints = EventTerm(
            func=mdp.reset_joints_to_sitting,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]),
            },
        )


    reset_robot_base = EventTerm(
        func = mdp.reset_root_state_uniform, 
        mode='reset', 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
 ]),
  "pose_range": {
             "x": (0.0, 0.0),
             "y": (0.0, 0.0),
             "z": (-0.20, -0.20),  
             "roll": (0.0, 0.0),
             "pitch": (-0.0, -0.0),  
             "yaw": (-0.0, 0.0),
         },
         "velocity_range": {
             "x": (0.0, 0.0),
             "y": (0.0, 0.0),
             "z": (0.0, 0.0),
             "roll": (0.0, 0.0),
             "pitch": (0.0, 0.0),
             "yaw": (0.0, 0.0),
         }
    }
    )

#     reset_robot_to_stand = EventTerm(
#     func=mdp.reset_robot_standing_pose,
#     mode="reset",
#     params={"asset_cfg": SceneEntityCfg("robot")},
# )





@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1)
    # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=0)
    # (3) Primary task: keep robot upright
    robot_pos = RewTerm(
        func=mdp.stand_upright,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    reward_lift_up_linear=RewTerm(
        func=mdp.reward_lift_up_linear,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_early_training,
        weight = -0.03
    )
    smooth_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001
    )
    smooth_acc= RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-07
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10
    )
    feet_clearance_cmd_linear = RewTerm(
        func=mdp.feet_clearance_cmd_linear,
        weight=-300
    )
    feet_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.4
    )

    foot_shift = RewTerm(
        func=mdp.foot_shift,
        weight=-50
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!RL_foot$|RR_foot$).*"), "threshold": 0.1},
    )
    action_q_diff = RewTerm(
        func=mdp.action_q_diff,
        weight = -1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01
    )






#adjust if moved to rough terrain
    # stay_up = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight = 4,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.35},
    # )
    # (4) Shaping tasks: slower joint movement
#     smooth_motion = RewTerm(
#         func=mdp.joint_vel_l1,
#         weight= 0.25,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
#     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
#     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
#     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
#     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
# ])},
#     )

    # no_bobbing = RewTerm(
    #     func=mdp.lin_vel_z_l2,
    #     weight = -.20,
    #     params={"asset_cfg": SceneEntityCfg("robot")})

    # no_turning = RewTerm(
    #     func=mdp.ang_vel_xy_l2,
    #     weight = -.20,
    #     params={"asset_cfg": SceneEntityCfg("robot")
    # })

    # remain_still = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight = -1.00,
    #     params={"asset_cfg": SceneEntityCfg("robot")})

    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(base|FL_.*|FR_.*|Head_.*)"), "threshold": 1.0},
    # )

    # undesired_contacts = RewTerm(
    #         func=mdp.undesired_contacts,
    #         weight=-1,
    #         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(RL_calf|RR_calf|RL_thigh|RR_thigh|RL_hip|RR_hip)"), "threshold": 1.0},
    #     )


    # desired_contacts = RewTerm(
    #     func=mdp.desired_contacts,
    #     weight=0.5,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(RL_foot|RR_foot)"), "threshold": 1.0},
    # )

    # desired_contacts = RewTerm(
    #     func=mdp.desired_contacts,
    #     weight=1,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(RL_foot|RR_foot|FL_foot|FR_foot)"), "threshold": 1.0},
    # )

#     base_contact_penalty = RewTerm(
#         func=mdp.illegal_contact,
#         weight=-0.3,
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
# )

#     head_contact_penalty = RewTerm(
#         func=mdp.illegal_contact,
#         weight=-0.5,
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head_.*"), "threshold": 1.0},
# )
    # flat_orientation_l2 = RewTerm(
    #     func=mdp.flat_orientation_l2,
    #     weight=-1,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # (5) Shaping tasks: maintaining orientation
    # orientation = RewTerm(
    #     func=mdp.balancing_on_four,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )

    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.3)
#     stand_pose_reward = RewTerm(
#         func=mdp.joint_pose_deviation_l2,
#         weight=2,
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#             "target": {
#     "FL_hip_joint":  0.1,  # small outward angle
#     "FL_thigh_joint": 0.9,  # upward
#     "FL_calf_joint":  -1.6,  # downward

#     "FR_hip_joint": -0.1,  # mirrored outward
#     "FR_thigh_joint": 0.9,
#     "FR_calf_joint": -1.6,

#     "RL_hip_joint":  0.1,
#     "RL_thigh_joint": 0.9,
#     "RL_calf_joint": -1.6,

#     "RR_hip_joint": -0.1,
#     "RR_thigh_joint": 0.9,
#     "RR_calf_joint": -1.6
# }
#         },
#     )

    # upright_pitch_reward = RewTerm(
    #     func=mdp.reward_upright_pitch,
    #     weight=1.0,  # adjust based on importance
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # feet_airtime = RewTerm(
    #     func=mdp.reward_feet_air_time_simple,
    #     weight=0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    

    # joint_pos_limit = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-1,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "num_steps": 0}
    # )

    # joint_vel_limit = RewTerm(
    #     func=mdp.joint_vel_limits,
    #     weight=-1,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "soft_ratio": 0.9}
    # )

#     stabilize_after_reset = RewTerm(
#         func=mdp.reward_foot_shift,
#         weight=-0.25,
#         params={"asset_cfg": SceneEntityCfg("robot")}
#     )

#     rear_air_reward = RewTerm(
#         func=mdp.reward_rear_air,
#         weight=1,
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
#         "RL_foot", "RR_foot", "RL_thigh", "RR_thigh"
#     ]), "num_steps": 1500}
# )
    # gait = RewTerm(
    #     func=mdp.reward_feet_clearance,
    #     weight=1,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )

    # reward_pitch = RewTerm(
    #     func=mdp.pitch,
    #     weight=2.0,  # adjust based on importance
    #     params={"asset_cfg": SceneEntityCfg("robot"),"sensor_cfg": SceneEntityCfg("contact_forces") , "num_steps": [2000, 4000, 10000]},
    # )
    
    # reward_height = RewTerm(
    #     func=mdp.height,
    #     weight=2.0,  # adjust based on importance
    #     params={"asset_cfg": SceneEntityCfg("robot"),"sensor_cfg": SceneEntityCfg("contact_forces"), "num_steps": [2000, 4000, 10000]},
    # )

    # reward_stand_attempt = RewTerm(
    #     func=mdp.reward_stand_air,
    #     weight=1.0,  # adjust based on importance
    #     params={"asset_cfg": SceneEntityCfg("robot"),"sensor_cfg": SceneEntityCfg("contact_forces")},
    # )
        
#     undesired_contact_all_but_feet = RewTerm(
#         func=mdp.undesired_all_but_feet,
#         weight=-10,
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!RL_foot$|RR_foot$).*")
# , "threshold": 1.0, "num_steps": 0},
#     )

#     undesired_contact_hands = RewTerm(
#         func=mdp.undesired_hands,
#         weight=-1.0,
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names = "FL_foot|FR_foot"), "threshold": 1.0, "num_steps": 2000},
#     )

#     undesired_contact_butt = RewTerm(
#         func=mdp.undesired_butt,
#         weight=-1.0,
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names = [
#     "RL_calf", "RR_calf"
# ]), "threshold": 1.0, "num_steps": 5000},
#     )

#     undesired_contact_thigh = RewTerm(
#         func=mdp.undesired_butt,
#         weight=-1.0,
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names = [
#     "RL_thigh", "RR_thigh", "RL_hip", "RR_hip"
# ]), "threshold": 1.0, "num_steps": 300},
#     )
    # reward_gait = RewTerm(
    #     func=mdp.gait,
    #     weight=0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "num_steps": 4000},
    # )

    # reward_thigh_extension = RewTerm(
    #     func=mdp.reward_thigh_extension,
    #     weight=2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"),"sensor_cfg": SceneEntityCfg("contact_forces"), "num_steps": 200},
    # )

    


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    illegal_with_mercy = DoneTerm(
        func=mdp.illegal_contacts_with_mercy,
        params={
            "sensor_cfg": BAD_CONTACTS,           # from step 1
            "allow_sensor_cfg": ALLOWED_EARLY,    # from step 1
            "threshold": 1.0,                     # contact force magnitude to count as 'hit'
            "allow_steps": 100,                    # mercy window length
        },
    )

    joint_limit_hit = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bounds": (-3.0, 3.0),  
        },
    )



    # (2) Cart out of boundshttps://sites.google.com/view/%20bipedal-motions-quadruped.
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )

    
    # fell_over = DoneTerm(
    #     func=mdp.base_too_low,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "num_steps":0}
    # )



    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )

    # illegal = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(base|FL_.*|FR_.*|Head_.*)"), "threshold": 3.0},
    # )


##
# Environment configuration
##

# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""
#     height = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "reward_height", "weight": 2, "num_steps": 4500}
#     )

#     pitch = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "reward_pitch", "weight": 4, "num_steps": 4500}
#     )

#     joint_pos = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "joint_pos_limit", "weight": -0.1, "num_steps": 3000}
#     )

#     joint_vel = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "joint_vel_limit", "weight": -0.1, "num_steps": 3000}
#     )
    
#     undesired = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "undesired_contact_all_but_feet", "weight": -1, "num_steps": 3500}
#     )

    # terminating = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "terminating", "weight": -1, "num_steps": 3500}
    # )
    
    


    
    # feet_off = CurrTerm(
    #     func=mdp.reward_rear_air
    # )


@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=1, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        # self.decimation = 2
        # self.episode_length_s = 20
        # # viewer settings
        # self.viewer.eye = (8.0, 0.0, 5.0)
        # # simulation settings
        # self.sim.dt = 1 / 100
        # self.sim.render_interval = self.decimation

        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120 
        self.sim.render_interval = self.decimation



#Ask about the difference between hyperparameters and whether that makes a difference
#Ask about the fact that they mention reward for pitch in paper but seemingly don't have it in code
#Maybe we need to train it to sit on 4 legs first, then use that model to train to stand up. 

#The paper explains each frame consists of joint positions,
# the orientation of the robot base, the last applied actions, the
# desired linear and angular velocity of the robot base, and
# the desired positions of front toes in the base coordination


#Spawn the robot in a specific knees bent low position

#Reduce jiteriness, perhaps through Td, Scale, or physics steps