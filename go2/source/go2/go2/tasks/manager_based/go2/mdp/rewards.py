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

    # Get base height
    height = env.scene[asset_cfg.name].data.root_pos_w[:, 2]  # z coordinate

    # Debug print — only from the first env to avoid spam
    print("Pitch (deg):", torch.rad2deg(pitch[0]).item())
    print("Height (m):", height[0].item())

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

    return reward


