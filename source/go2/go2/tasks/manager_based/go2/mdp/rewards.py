# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.math import euler_xyz_from_quat
import math; 

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
_debug_counter = 0


# def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint position deviation from a target value."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # wrap the joint positions to (-pi, pi)
#     joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
#     # compute the reward
#     return torch.sum(torch.square(joint_pos - target), dim=1)

def stay_upright(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    z_pos = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    diff = torch.abs(target - z_pos)
    reward = 1 - diff

    print(env.scene["robot"].data.root_pos_w[:, 2])

    return reward


def balancing_on_four(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return 0.1 * torch.ones(env.num_envs, device=env.device)



# def base_too_low(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     z_pos = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
#     terminated = z_pos < 0.10
#     return terminated.bool()

# def standing_pose(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     return 

def joint_pose_deviation_l2(env, asset_cfg: SceneEntityCfg, target: dict):
    asset = env.scene[asset_cfg.name]
    joint_names = list(target.keys())
    target_values = torch.tensor([target[name] for name in joint_names], device=env.device)
    
    joint_ids = asset_cfg.joint_ids
    joint_pos = asset.data.joint_pos[:, joint_ids]
    #print(joint_pos)

    error = joint_pos - target_values
    reward = torch.exp(-0.5 * torch.norm(error, dim=-1))
    #print(reward)
    return reward


def base_too_low(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, grace_steps=60):
    
    z_pos = env.scene[asset_cfg.name].data.root_pos_w[:, 2]

    grace_over = env.episode_length_buf >= grace_steps

    too_low = z_pos < 0.15

    terminated = torch.logical_and(grace_over, too_low)
    print(z_pos)
    return terminated

def reward_upright_pitch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    quat = env.scene[asset_cfg.name].data.root_quat_w
    _, pitch, _ = euler_xyz_from_quat(quat)

    TARGET_PITCH = -math.pi / 3
    std_dev = 0.20

    reward = torch.exp(-(pitch - TARGET_PITCH)**2 / (2 * std_dev**2))

    if hasattr(env, "logger"):
        env.logger.log_scalar("reward/upright_pitch", reward.mean().item())
   
    return reward

def reward_feet_air_time_simple(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene["contact_forces"]  # make sure this is the correct name

    # Hind foot names
    foot_names = ["RL_foot", "RR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_names]

    # Get vertical velocities
    foot_vel_z = asset.data.body_vel_w[:, foot_ids, 2]

    # Get contact forces (net contact forces in world frame)
    contact_z = contact_sensor.data.net_forces_w[:, foot_ids, 2]  # ✅ corrected line
    contact = contact_z > 1.0

    impact = (foot_vel_z < -0.5) & contact
    reward = torch.sum(impact.float(), dim=1) * 0.5

# 👇 Add this to log to TensorBoard (if enabled)
    if hasattr(env, "logger"):
        env.logger.log_scalar("reward/feet_air_time_simple", reward.mean().item())
    return reward

def reward_lift_up_linear(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Linearly reward lifting the base above a minimum height.

    Reward = 0 below min height  
    Reward ramps from 0 → 1 between min and max height  
    Reward = 1 at or above max height
    """
    # Get base z position
    z_pos = env.scene[asset_cfg.name].data.root_pos_w[:, 2]

    # Define shaping thresholds (you can tune these)
    min_height = 0.20  # reward starts here
    max_height = 0.34  # full reward at this point

    # Normalize reward linearly in [min_height, max_height]
    reward = (z_pos - min_height) / (max_height - min_height)
    reward = torch.clamp(reward, 0.0, 1.0)

# 👇 Add this to log to TensorBoard (if enabled)
    if hasattr(env, "logger"):
        env.logger.log_scalar("reward/lift_up_linear", reward.mean().item())
    return reward

def reward_foot_shift(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalizes early foot movement after reset to encourage stable, planted feet.
    Higher reward when feet stay near initial positions. Only applies during early steps.
    """
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w  # shape: (num_envs, num_bodies, 3)

    # Get foot indices
    front_ids = [asset.body_names.index(name) for name in ["FL_foot", "FR_foot"]]
    rear_ids  = [asset.body_names.index(name) for name in ["RL_foot", "RR_foot"]]

    # Save initial foot positions if not already done
    if getattr(env, "_init_foot_pos", None) is None:
        env._init_foot_pos = torch.clone(foot_pos)

    init_foot_pos = env._init_foot_pos

    # Compute per-foot displacement in XY plane
    front_shift = torch.norm(foot_pos[:, front_ids, :2] - init_foot_pos[:, front_ids, :2], dim=-1).mean(dim=1)
    rear_shift  = torch.norm(foot_pos[:, rear_ids, :2]  - init_foot_pos[:, rear_ids, :2],  dim=-1).mean(dim=1)

    # Only apply during early steps
    grace_steps = 50
    condition = (env.episode_length_buf < grace_steps).float()

    # Penalize total shift
    reward = -(front_shift + rear_shift) * condition

# 👇 Add this to log to TensorBoard (if enabled)
    if hasattr(env, "logger"):
        env.logger.log_scalar("reward/reward_foot_shift", reward.mean().item())
    return reward


# def reward_rear_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """
#     Encourages rear feet to stay in contact and penalizes rearing behavior 
#     where the rear feet are in the air but the calves are contacting the ground.

#     Aims to catch the undesirable situation of the robot tipping backward.
#     """
#     contact_forces = env.scene[sensor_cfg.name].data.net_forces_w  # (num_envs, num_bodies, 3)

#     # Index rear feet and rear calves
#     rear_foot_names = ["RL_foot", "RR_foot"]
#     rear_calf_names = ["RL_calf", "RR_calf"]
#     body_names = sensor_cfg.body_names if sensor_cfg.body_names else env.scene[sensor_cfg.name].body_names

#     rear_foot_ids = [body_names.index(name) for name in rear_foot_names]
#     rear_calf_ids = [body_names.index(name) for name in rear_calf_names]

#     # Contact = z-force < threshold => not in contact
#     foot_in_air = contact_forces[:, rear_foot_ids, 2] < 1.0   # shape (num_envs, 2)
#     calf_in_contact = contact_forces[:, rear_calf_ids, 2] >= 1.0

#     # Unhealthy: calves touching ground, but feet not
#     unhealthy = torch.logical_and(calf_in_contact, foot_in_air)

#     # Reward is:
#     # +1 if both feet are in contact
#     # +penalty if unhealthy case is triggered
#     feet_contact = torch.all(~foot_in_air, dim=1).float()
#     unhealthy_penalty = unhealthy.sum(dim=1).float()  # 0 to 2

#     # grace_steps = 50
#     # active = (env.episode_length_buf >= grace_steps).float()
#     # reward = (feet_contact - unhealthy_penalty) * active    
#     reward = (feet_contact - unhealthy_penalty)

#     return reward
def reward_rear_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Encourages rear feet to stay in contact and penalizes rearing behavior 
    where the rear feet are in the air but the rear legs (hips, thighs, calves) are contacting the ground.

    Aims to catch the undesirable situation of the robot tipping backward.
    """
    contact_forces = env.scene[sensor_cfg.name].data.net_forces_w  # (num_envs, num_bodies, 3)

    # Index rear feet and rear leg segments
    rear_foot_names = ["RL_foot", "RR_foot"]
    rear_leg_names = ["RL_calf", "RR_calf", "RL_thigh", "RR_thigh", "RL_hip", "RR_hip"]
    body_names = sensor_cfg.body_names if sensor_cfg.body_names else env.scene[sensor_cfg.name].body_names

    rear_foot_ids = [body_names.index(name) for name in rear_foot_names]
    rear_leg_ids = [body_names.index(name) for name in rear_leg_names]

    # Determine contact states
    foot_in_air = contact_forces[:, rear_foot_ids, 2] < 1.0               # shape (num_envs, 2)
    leg_in_contact = contact_forces[:, rear_leg_ids, 2] >= 1.0            # shape (num_envs, 6)

    # Unhealthy if any leg segment is touching while corresponding foot is off the ground
    # For simplicity, treat all leg contacts as penalizing regardless of alignment
    unhealthy = torch.any(leg_in_contact, dim=1) & torch.any(foot_in_air, dim=1)

    # Compute reward
    feet_contact = torch.all(~foot_in_air, dim=1).float()
    unhealthy_penalty = unhealthy.float()

    grace_steps = 60
    active = (env.episode_length_buf >= grace_steps).float()
    reward = (feet_contact - (unhealthy_penalty *1.5)) * active  
    #i added a *1.5 just to make the penalty weigh more.  

    return reward

def reward_feet_clearance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    
    asset = env.scene[asset_cfg.name]

    # Get rear foot indices
    rear_foot_names = ["RL_foot", "RR_foot"]
    foot_ids = [asset.body_names.index(name) for name in rear_foot_names]

    # Get foot height and XY positions
    foot_heights = asset.data.body_pos_w[:, foot_ids, 2]  # (num_envs, 2)
    foot_xy = asset.data.body_pos_w[:, foot_ids, :2]      # (num_envs, 2, 2)

    # Fake motion phase for now — replace with real gait phase if available
    # Range: [0, 1], peak = mid-swing
    phase = torch.abs(torch.sin(env.episode_length_buf.unsqueeze(1) / 30.0 * math.pi))  # (num_envs, 1)
    phase = phase.repeat(1, 2)  # Repeat for both feet

    # Terrain height under feet
    # terrain_heights = env.scene["terrain"].get_heights_at(foot_xy)  # assumes API exists like Isaac Gym
    terrain_heights = 0
    target_clearance = 0.05  # target foot clearance in swing

    target_heights = target_clearance * phase + terrain_heights + 0.02  # (num_envs, 2)

    # Optional: mask out feet that should be in contact
    desired_contact = env.scene["contact_states"].data[asset_cfg.name][:, foot_ids]  # (num_envs, 2)
    in_swing = 1.0 - desired_contact.float()  # (1 - contact) = swing phase

    # Compute reward
    clearance_error = torch.square(target_heights - foot_heights)
    reward = -clearance_error * in_swing  # penalize low feet during swing

    # Apply after grace period
    grace_steps = 50
    active = (env.episode_length_buf >= grace_steps).float().unsqueeze(1)
    reward = reward * active

    # Return scalar per env
    return reward.sum(dim=1)
