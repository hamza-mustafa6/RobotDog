
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn,VecEnvObs
from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from collections.abc import Sequence
from typing import Any, ClassVar
import torch

from sympy.physics.units import frequency


class go2_env(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # self.num_envs = cfg.scene.num_envs
        self.dt = cfg.sim.dt
        self._init_buffers()


    def _init_buffers(self):
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device='cuda:0', requires_grad=False)
        self.default_dof_pos = torch.zeros(12, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = self.scene['robot'].data.joint_pos[:].clone()
        # for i in range(12):
        #     name = self.dof_names[i]
        #     angle = self.cfg.init_state.default_joint_angles[name]
        #     self.default_dof_pos[i] = angle
        #     init_angle = self.cfg.init_state.init_joint_angles[name]
        #     self.init_dof_pos[i] = init_angle
        #     self.init_dof_pos_range[i][0] = self.cfg.init_state.init_joint_angles_range[name][0]
        #     self.init_dof_pos_range[i][1] = self.cfg.init_state.init_joint_angles_range[name][1]
        #     found = False
        #     for dof_name in self.cfg.control.stiffness.keys():
        #         if dof_name in name:
        #             self.p_gains[i] = self.cfg.control.stiffness[dof_name]
        #             self.d_gains[i] = self.cfg.control.damping[dof_name]
        #             found = True
        #     if not found:
        #         self.p_gains[i] = 0.
        #         self.d_gains[i] = 0.
        #         if self.cfg.control.control_type in ["P", "V"]:
        #             print(f"PD gain of joint {name} were not defined, setting them to zero")



    def pre_physics_step(self):
        super().pre_physics_step()
        # Inject your custom logic here

    def post_physics_step(self):
        super().post_physics_step()
        # Custom post-processing
        self.last_dof_pos[:] = self.scene['robot'].data.joint_pos[:].clone()

    # def _reward_custom(self):
    #     # Example of custom reward
    #     return torch.ones(self.num_envs, device=self.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))



        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)

        self.common_step_counter += 1  # total step (common for all envs)

        # post physics call back
        self._step_contact_targets()
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.last_dof_pos[:] = self.scene['robot'].data.joint_pos[:].clone()
        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _step_contact_targets(self):
        # TODO: fill in reasonable numbers
        # frequencies = 3 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        frequencies =2.5
        phases = 0.5
        offsets = 0
        bounds = 0

        durations = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)  # 0.5
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)


        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                    0.5 / (1 - durations[swing_idxs]))

        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _get_heights_at_points(self, points):
        """ Get vertical projected terrain heights at points
        points: a tensor of size (num_envs, num_points, 2) in world frame
        """
        points = points.clone()
        num_points = points.shape[1]

        return torch.zeros(self.num_envs, num_points, device=self.device, requires_grad=False)


    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:
        """Resets the specified environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset the specified environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self._reset_idx(env_ids)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute()

        feet_indices = torch.tensor([15, 16, 17, 18], dtype=torch.int64, device='cuda:0')
        self.init_feet_positions = self.scene['robot'].data.body_state_w[:, feet_indices, 0:3].clone()
        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        # return observations
        return self.obs_buf, self.extras


 