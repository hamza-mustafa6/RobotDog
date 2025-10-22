from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def prev_action(env: ManagerBasedEnv) -> torch.Tensor:
    return env.action_manager.prev_action
    

def reset_joints_to_sitting(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    
    #Resets thighs to sit
    joint_pos[:, 4] = 1.3
    joint_pos[:, 5] = 1.3
    joint_pos[:, 6] = 1.4
    joint_pos[:, 7] = 1.4

    #Resets calves to sit
    joint_pos[:, 8] = -2.4
    joint_pos[:, 9] = -2.4
    joint_pos[:, 10] = -2.4
    joint_pos[:, 11] = -2.4

    joint_names = asset.data.joint_names          # list of joint names

    # print(joint_names)
    # print(joint_pos)
        
    
    # joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    # joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)



    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
  
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
