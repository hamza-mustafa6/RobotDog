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
from isaaclab.utils.math import quat_apply, yaw_quat, quat_apply, quat_apply_yaw, quat_conjugate
import numpy as np
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


def stand_upright(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float, device='cuda:0').repeat(env.num_envs, 1)
    upright_vec = torch.tensor([0.2, 0., 1.0], dtype=torch.float, device='cuda:0').repeat(env.num_envs, 1)
    forward = quat_apply(asset.data.root_quat_w, forward_vec)
    upright_vec = quat_apply_yaw(asset.data.root_quat_w, upright_vec)
    cosine_dist = torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)
    reward = torch.square(0.5 * cosine_dist + 0.5)
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
    min_height = 0.15  # reward starts here
    max_height = 0.42  # full reward at this point

    # Normalize reward linearly in [min_height, max_height]
    reward = (z_pos - min_height) / (max_height - min_height)
    reward = torch.clamp(reward, 0.0, 1.0)

# 👇 Add this to log to TensorBoard (if enabled)
    if hasattr(env, "logger"):
        env.logger.log_scalar("reward/lift_up_linear", reward.mean().item())
    return reward

def action_rate_l2_early_training(env) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""

    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def feet_clearance_cmd_linear(env,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    phases = 1 - torch.abs(1.0 - torch.clip((env.foot_indices[:, -2:] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    feet_indices = torch.tensor([15,16,17,18], dtype=torch.int64, device='cuda:0')
    foot_positions = asset.data.body_state_w[:, feet_indices, 0:3]
    foot_height = (foot_positions[:, -2:, 2]).view(env.num_envs, -1)# - reference_heights
    terrain_at_foot_height = env._get_heights_at_points(foot_positions[:, -2:, :2])
    target_height = 0.05 * phases + terrain_at_foot_height + 0.02
    rew_foot_clearance = torch.square(target_height - foot_height) * (1 - env.desired_contact_states[:, -2:])
    condition = env.episode_length_buf > 30
    rew_foot_clearance = rew_foot_clearance * condition.unsqueeze(dim=-1).float()
    rew_foot_clearance = rew_foot_clearance
    return torch.sum(rew_foot_clearance, dim=1)

def feet_slip(env,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    feet_indices = torch.tensor([15, 16, 17, 18], dtype=torch.int64, device='cuda:0')
    foot_positions = asset.data.body_state_w[:, feet_indices, 0:3]
    foot_velocities = asset.data.body_state_w[:, feet_indices, 7:10]  # shape: (num_envs, num_bodies, 13)
    foot_velocities_ang = asset.data.body_state_w[:, feet_indices, 10:13]
    condition = foot_positions[:, :, 2] < 0.03
    # xy lin vel
    foot_velocities = torch.square(torch.norm(foot_velocities[:, :, 0:2], dim=2).view(env.num_envs, -1))
    # yaw ang vel
    foot_ang_velocities = torch.square(torch.norm(foot_velocities_ang[:, :, 2:] / np.pi, dim=2).view(env.num_envs, -1))
    rew_slip = torch.sum(condition.float() * (foot_velocities + foot_ang_velocities), dim=1)
    return rew_slip

def reward_foot_shift(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalizes early foot movement after reset to encourage stable, planted feet.
    Higher reward when feet stay near initial positions. Only applies during early steps.
    """
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w  # shape: (num_envs, num_bodies, 3)

    # Get foot indices
    front_ids = [asset.body_names.index(name) for name in ["FL_foot", "FR_foot"]]
    rear_ids = [asset.body_names.index(name) for name in ["RL_foot", "RR_foot"]]

    # Save initial foot positions if not already done
    if getattr(env, "_init_foot_pos", None) is None:
        env._init_foot_pos = torch.clone(foot_pos)

    init_foot_pos = env._init_foot_pos

    # Compute per-foot displacement in XY plane
    # front_shift = torch.norm(foot_pos[:, front_ids, :2] - init_foot_pos[:, front_ids, :2], dim=-1).mean(dim=1)
    rear_shift = torch.norm(foot_pos[:, rear_ids, :2] - init_foot_pos[:, rear_ids, :2], dim=-1).mean(dim=1)

    # Only apply during early steps
    grace_steps = 50
    condition = (env.episode_length_buf < grace_steps).float()

    # Penalize total shift
    reward = -(rear_shift) * condition

    # 👇 Add this to log to TensorBoard (if enabled)
    if hasattr(env, "logger"):
        env.logger.log_scalar("reward/reward_foot_shift", reward.mean().item())
    print(reward)
    return reward

def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment

    reward = torch.sum(is_contact, dim=1)
    cond = env.episode_length_buf > 30
    reward = reward * cond.float()
    return reward

def action_q_diff(env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    condition = env.episode_length_buf <= 30
    actions = torch.clip(env.action_manager.action, -100, 100)
    q_diff_buf = torch.abs(asset.data.default_joint_pos + 0.25 * actions - asset.data.joint_pos)
    reward = torch.sum(torch.square(q_diff_buf), dim=-1) * condition.float()
    return reward

def applied_torque_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # # TODO: We need to fix this to support implicit joints.
    # condition = env.episode_length_buf <= 30
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    torque_limits_penalty = torch.clamp(torch.sum(out_of_limits, dim=1), max=10.0)
    return torque_limits_penalty

def foot_shift(env,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    feet_indices = torch.tensor([15, 16, 17, 18], dtype=torch.int64, device='cuda:0')
    desired_foot_positions = torch.clone(env.init_feet_positions[:, 2:])
    desired_foot_positions[:, :, 2] = 0.02
    foot_positions = asset.data.body_state_w[:, feet_indices, 0:3]
    rear_foot_shift = torch.norm(foot_positions[:, 2:] - desired_foot_positions, dim=-1).mean(dim=1)
    init_ffoot_positions = torch.clone(env.init_feet_positions[:, :2])
    front_foot_shift = torch.norm( torch.stack([
            (init_ffoot_positions[:, :, 0] - foot_positions[:, :2, 0]).clamp(min=0),
            torch.abs(init_ffoot_positions[:, :, 1] - foot_positions[:, :2, 1])
        ], dim=-1), dim=-1).mean(dim=1)
    condition = env.episode_length_buf < 30
    reward = (front_foot_shift + rear_foot_shift) * condition.float()
    return reward


def illegal_contacts_with_mercy(
    env,
    sensor_cfg: SceneEntityCfg,           # BAD_CONTACTS
    allow_sensor_cfg: SceneEntityCfg,     # ALLOWED_EARLY
    threshold: float = 1.0,
    allow_steps: int = 30,
):
    """
    Returns a boolean [num_envs] that is True when an environment should reset
    due to illegal contacts, except during an early 'mercy' window where
    contacts on allowed bodies are forgiven.
    """
    # Grab both views of the same ContactSensor (filtered by the SceneEntityCfg)
    bad = env.scene[sensor_cfg.name]
    good = env.scene[allow_sensor_cfg.name]

    # NOTE: the manager resolves body filters in SceneEntityCfg for us.
    # So we can safely index with the resolved ids on the sensor’s data buffer.
    # If the sensor has N bodies in the filtered view, this already matches.
    bad_forces = bad.data.net_forces_w  # [num_envs, N_bad, 3]
    allowed_forces = good.data.net_forces_w  # [num_envs, N_allow, 3]

    bad_hit = torch.any(torch.norm(bad_forces, dim=-1) > threshold, dim=1)  # [num_envs]
    allow_hit = torch.any(torch.norm(allowed_forces, dim=-1) > threshold, dim=1)  # [num_envs]

    # Early-step mercy window
    in_mercy = env.episode_length_buf <= allow_steps  # [num_envs]

    # Reset if: bad contact AND NOT(allowed contact during mercy)
    # should_reset = torch.logical_and(bad_hit, torch.logical_not(torch.logical_and(allow_hit, in_mercy)))
    should_reset = torch.logical_and(bad_hit, torch.logical_not(in_mercy))

    return should_reset















def balancing_on_four(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return 0.1 * torch.ones(env.num_envs, device=env.device)


def base_too_low(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, num_steps: int) -> torch.Tensor:
    step = env.common_step_counter
    z_pos = env.scene[asset_cfg.name].data.root_pos_w[:, 2]  # shape: (num_envs,)

    # tensor of shape (num_envs,)
    under_height = z_pos < 0.15

    # Grace period: skip termination if we're still in the first 50 steps of episode
    grace_steps = 3
    condition = env.episode_length_buf < grace_steps  # shape: (num_envs,)
    
    # Only allow termination *after* grace period ends
    terminated = torch.logical_and(~condition, under_height)  # shape: (num_envs,)

    return terminated


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

def reward_upright_pitch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    quat = env.scene[asset_cfg.name].data.root_quat_w
    _, pitch, _ = euler_xyz_from_quat(quat)

    TARGET_PITCH = -math.pi / 2
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
#     return reward
def reward_rear_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, num_steps: int) -> torch.Tensor:
    """
    Encourages rear feet to stay in contact and penalizes rearing behavior 
    where the rear feet are in the air but the rear legs (hips, thighs, calves) are contacting the ground.

    Aims to catch the undesirable situation of the robot tipping backward.
    """
    contact_forces = env.scene[sensor_cfg.name].data.net_forces_w  # (num_envs, num_bodies, 3)

    # Index rear feet and rear leg segments
    rear_foot_names = ["RL_foot", "RR_foot"]
    rear_leg_names = ["RL_thigh", "RR_thigh"]
    body_names = sensor_cfg.body_names if sensor_cfg.body_names else env.scene[sensor_cfg.name].body_names

    rear_foot_ids = [body_names.index(name) for name in rear_foot_names]
    rear_leg_ids = [body_names.index(name) for name in rear_leg_names]

    # Determine contact states
    foot_in_air = contact_forces[:, rear_foot_ids, 2] < 1               # shape (num_envs, 2)
    leg_in_contact = contact_forces[:, rear_leg_ids, 2] >= 1.0            # shape (num_envs, 6)

    # Unhealthy if any leg segment is touching while corresponding foot is off the ground
    # For simplicity, treat all leg contacts as penalizing regardless of alignment
    unhealthy = torch.any(leg_in_contact, dim=1) & torch.any(foot_in_air, dim=1)

    # Compute reward
    feet_contact = torch.any(~foot_in_air, dim=1).float()
    unhealthy_penalty = unhealthy.float()

    step = env.common_step_counter
    # grace_steps = 90
    active = torch.full((env.num_envs,), float(step >= num_steps), device=env.device)
    reward = ((feet_contact) - (unhealthy_penalty)) * active  

    return reward

def reward_feet_clearance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Encourage rear foot clearance during swing phase using a fake gait phase.
    """
    asset = env.scene[asset_cfg.name]
    contact_forces = env.scene["contact_forces"].data.net_forces_w  # (num_envs, num_bodies, 3)

    rear_foot_names = ["RL_foot", "RR_foot"]
    foot_ids = [asset.body_names.index(name) for name in rear_foot_names]

    # Get foot heights and XY positions
    foot_heights = asset.data.body_pos_w[:, foot_ids, 2]  # (num_envs, 2)
    phase = torch.abs(torch.sin(env.episode_length_buf.unsqueeze(1) / 30.0 * math.pi))
    phase = phase.repeat(1, 2)

    # Terrain height (flat ground assumed)
    terrain_heights = 0.0
    target_clearance = 0.05
    target_heights = target_clearance * phase + terrain_heights + 0.02

    # Determine if foot is in contact (z-force > threshold means contact)
    contact_thresh = 1.0
    contact = contact_forces[:, foot_ids, 2] > contact_thresh  # shape: (num_envs, 2)
    swing_phase = ~contact  # True if not in contact

    # Penalize low feet during swing
    clearance_error = torch.square(target_heights - foot_heights)
    reward = -clearance_error * swing_phase.float()

    # Apply grace period
    grace_steps = 80
    active = (env.episode_length_buf >= grace_steps).float().unsqueeze(1)
    reward = reward * active

    # Return scalar per env
    return reward.sum(dim=1)

def reward_stand_air(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward for standing upright *and* having rear feet off the ground slightly.
    - Only applies during early part of the episode (< contact_steps).
    """

    base_quat = env.scene[asset_cfg.name].data.root_quat_w
    forward_vec = torch.tensor([0, 0, 1.0], device=env.device).expand(env.num_envs, -1)
    z_proj = quat_apply(base_quat, forward_vec)[:, 2]  # how upright the body is

    # Get rear foot positions in world space
    body_names = env.scene[sensor_cfg.name].body_names
    rear_feet = ["RL_foot", "RR_foot"]
    rear_ids = [body_names.index(name) for name in rear_feet]
    rear_pos = env.scene[asset_cfg.name].data.body_pos_w[:, rear_ids, 2]  # z positions of rear feet

    grace_steps = 100
    grace = (env.episode_length_buf < grace_steps).float()
    condition = torch.logical_and(
        grace,
        torch.logical_and(
            z_proj < 0.9,  # not fully vertical
            torch.any(rear_pos > 0.03, dim=1)  # feet lifted a bit
        )
    )

    return condition.float()


#Curriculum Learning:

def pitch(env: ManagerBasedRLEnv, num_steps: List[int], sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    step = env.common_step_counter
    quat = env.scene[asset_cfg.name].data.root_quat_w
    _, pitch, _ = euler_xyz_from_quat(quat)
    #scale = is_standing_on_hind_legs(env, asset_cfg, sensor_cfg)


    if step < num_steps[0]:
        TARGET_PITCH = -0.8
        std_dev = 0.25

        reward = torch.exp(-(pitch - TARGET_PITCH)**2 / (2 * std_dev**2))

    elif step < num_steps[1]:
        TARGET_PITCH = -1.3
        std_dev = 0.20

        reward = torch.exp(-(pitch - TARGET_PITCH)**2 / (2 * std_dev**2))

    elif step < num_steps[2]:
        TARGET_PITCH = -math.pi / 2
        std_dev = 0.20

        reward = torch.exp(-(pitch - TARGET_PITCH)**2 / (2 * std_dev**2))

    else:
        TARGET_PITCH = -math.pi / 2
        std_dev = 0.20
        reward = torch.exp(-(pitch - TARGET_PITCH)**2 / (2 * std_dev**2))

    # if hasattr(env, "logger"):
    #     env.logger.log_scalar("reward/upright_pitch", reward.mean().item())
    return reward #* scale
    

# def joint_pos_limits(env: ManagerBasedRLEnv,num_steps:int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize joint positions if they cross the soft limits.

#     This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
#     """
#     step = env.common_step_counter
#     reward = torch.zeros(env.num_envs, device=env.device)

#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # compute out of limits constraints
#     out_of_limits = -(
#         asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
#     ).clip(max=0.0)
#     out_of_limits += (
#         asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
#     ).clip(min=0.0)

#     if step > num_steps:
#         reward = torch.sum(out_of_limits, dim=1)
        
#     return reward



def height(env: ManagerBasedRLEnv, num_steps: List[int], sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    step = env.common_step_counter
    # Get base z position
    z_pos = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    #scale = is_standing_on_hind_legs(env, asset_cfg, sensor_cfg)

      
    if step < num_steps[0]:
        min_height = 0.20  
        max_height = 0.25

    elif step < num_steps[1]:
        min_height = 0.25  
        max_height = 0.30

    elif step < num_steps[2]:
        min_height = 0.30  
        max_height = 0.40
    else:
        min_height = 0.35  
        max_height = 0.40

    # Normalize reward linearly in [min_height, max_height]
    reward = (z_pos - min_height) / (max_height - min_height)
    reward = torch.clamp(reward, 0.0, 1.0)

    # if hasattr(env, "logger"):
    #     env.logger.log_scalar("reward/upright_pitch", reward.mean().item())
    return reward #* scale

def undesired_butt(env: ManagerBasedRLEnv, num_steps: int, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    step = env.common_step_counter

    reward = torch.zeros(env.num_envs, device=env.device)

    if step > num_steps:
        """Penalize undesired contacts as the number of violations that are above a threshold."""
        # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # check if contact force is above threshold
        net_contact_forces = contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        # sum over contacts for each environment
        reward = torch.sum(is_contact, dim=1)

    # if hasattr(env, "logger"):
    #     env.logger.log_scalar("reward/upright_pitch", reward.mean().item())
    return reward
       
def undesired_hands(env: ManagerBasedRLEnv, num_steps: int, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    step = env.common_step_counter

    reward = torch.zeros(env.num_envs, device=env.device)

    if step > num_steps:
        """Penalize undesired contacts as the number of violations that are above a threshold."""
        # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # check if contact force is above threshold
        net_contact_forces = contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        # sum over contacts for each environment
        reward = torch.sum(is_contact, dim=1)

    # if hasattr(env, "logger"):
    #     env.logger.log_scalar("reward/upright_pitch", reward.mean().item())
    return reward

def undesired_all_but_feet(env: ManagerBasedRLEnv, num_steps: int, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    step = env.common_step_counter

    reward = torch.zeros(env.num_envs, device=env.device)

    if step > num_steps:
        """Penalize undesired contacts as the number of violations that are above a threshold."""
        # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # check if contact force is above threshold
        net_contact_forces = contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        # sum over contacts for each environment
        reward = torch.sum(is_contact, dim=1)

    # if hasattr(env, "logger"):
    #     env.logger.log_scalar("reward/upright_pitch", reward.mean().item())
    return reward
       
def gait(env: ManagerBasedRLEnv, num_steps: int, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    step = env.common_step_counter
    reward = torch.zeros(env.num_envs, device=env.device)

    if step > num_steps:
        """
        Encourage rear foot clearance during swing phase using a fake gait phase.
        """
        asset = env.scene[asset_cfg.name]
        contact_forces = env.scene["contact_forces"].data.net_forces_w  # (num_envs, num_bodies, 3)

        rear_foot_names = ["RL_foot", "RR_foot"]
        foot_ids = [asset.body_names.index(name) for name in rear_foot_names]

        # Get foot heights and XY positions
        foot_heights = asset.data.body_pos_w[:, foot_ids, 2]  # (num_envs, 2)

        # Fake motion phase shaping — peaks at mid-swing
        phase = torch.abs(torch.sin(env.episode_length_buf.unsqueeze(1) / 30.0 * math.pi))
        phase = phase.repeat(1, 2)

        # Terrain height (flat ground assumed)
        terrain_heights = 0.0
        target_clearance = 0.05
        target_heights = target_clearance * phase + terrain_heights + 0.02

        # Determine if foot is in contact (z-force > threshold means contact)
        contact_thresh = 1.0
        contact = contact_forces[:, foot_ids, 2] > contact_thresh  # shape: (num_envs, 2)
        swing_phase = ~contact  # True if not in contact

        # Penalize low feet during swing
        clearance_error = torch.square(target_heights - foot_heights)
        reward = -clearance_error * swing_phase.float()

        # Apply grace period
        grace_steps = 80
        active = (env.episode_length_buf >= grace_steps).float().unsqueeze(1)
        reward = reward * active
        reward = reward.sum(dim=1)

        # Return scalar per env
    return reward

def reward_thigh_extension(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, num_steps: int) -> torch.Tensor:
    """
    Rewards the robot for extending its thigh joints (helps promote upright standing).

    Aims to encourage hip extension — a key aspect of standing posture.
    """
    step = env.common_step_counter
    reward = torch.zeros(env.num_envs, device=env.device)
    #scale = is_standing_on_hind_legs(env, asset_cfg, sensor_cfg)


    if step > num_steps:
        asset = env.scene[asset_cfg.name]

        # These are the hip-to-thigh joints
        thigh_joint_names = ["RL_thigh_joint", "RR_thigh_joint"]
        joint_names = asset.joint_names
        joint_ids = [joint_names.index(name) for name in thigh_joint_names]

        # Get joint positions
        joint_angles = asset.data.joint_pos[:, joint_ids]  # shape: (num_envs, 2)
        if step%50 == 0:
            print(joint_angles)

        # Define target extended angle (positive means backward)
        target_angle = 1.6  # tweak as needed based on your robot’s URDF

        # Use a Gaussian-shaped reward (peak at target_angle)
        std_dev = 0.3  # allow some slack
        diff = joint_angles - target_angle
        reward = torch.exp(-(diff**2) / (2 * std_dev**2))

        # Average over both thighs
        reward = reward.mean(dim=1) #* scale

    
    return reward 


def is_standing_on_hind_legs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                             height_thresh=0.33, pitch_target=-math.pi/2, pitch_tol=0.2) -> torch.Tensor:
    """
    Outputs a scaling factor ∈ [0, 1] indicating how well the robot is standing on hind legs.
    Considers height, pitch, and rear foot contact.
    """
    # --- Height factor ---
    base_z = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    height_score = torch.clamp((base_z - height_thresh) / (0.05 + 1e-5), 0.0, 1.0)

    # --- Pitch factor ---

    base_quat = env.scene[asset_cfg.name].data.root_quat_w
    _, pitch, _ = euler_xyz_from_quat(base_quat)
    pitch_diff = torch.abs(pitch - pitch_target)
    pitch_score = torch.clamp(1.0 - pitch_diff / pitch_tol, 0.0, 1.0)

    # --- Rear feet contact factor ---
    contact_forces = env.scene[sensor_cfg.name].data.net_forces_w
    body_names = env.scene[sensor_cfg.name].body_names
    rear_feet = ["RL_foot", "RR_foot"]
    rear_ids = [body_names.index(n) for n in rear_feet]
    feet_contact = (contact_forces[:, rear_ids, 2] > 1.0).float()  # (num_envs, 2)
    feet_score = feet_contact.mean(dim=1)  # (num_envs,)

    # Final dynamic scaling factor
    scaling = height_score * pitch_score * feet_score  # ∈ [0, 1]
    return scaling



# def modify_reward_weight(env: ManagerBasedRLEnv, num_steps: int):
#     """Curriculum that modifies a reward weight a given number of steps.

#     Args:
#         env: The learning environment.
#         env_ids: Not used since all environments are affected.
#         term_name: The name of the reward term.
#         weight: The weight of the reward term.
#         num_steps: The number of steps after which the change should be applied.
#     """
#     if env.common_step_counter > num_steps:
        

# def reset_robot_standing_pose(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
#     # Hardcoded joint positions for a bipedal standing pose
#     joint_targets = {
#         "FL_hip_joint":  0.3,
#         "FL_thigh_joint": 1.2,
#         "FL_calf_joint": -2.0,

#         "FR_hip_joint": -0.3,
#         "FR_thigh_joint": 1.2,
#         "FR_calf_joint": -2.0,

#         "RL_hip_joint":  0.0,
#         "RL_thigh_joint": 0.8,
#         "RL_calf_joint": -1.3,

#         "RR_hip_joint":  0.0,
#         "RR_thigh_joint": 0.8,
#         "RR_calf_joint": -1.3,
#     }

#     asset = env.scene[asset_cfg.name]
#     joint_indices = [asset.joint_names.index(j) for j in joint_targets.keys()]
#     joint_positions = torch.zeros((env.num_envs, len(joint_indices)), device=env.device)

#     for i, joint_name in enumerate(joint_targets.keys()):
#         joint_positions[:, i] = joint_targets[joint_name]

#     asset.set_joint_positions(joint_positions, joint_indices, env_ids)
