# Copyright (c) 2025, Vettar Project
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor

from .vettar_env_cfg import VettarFlatEnvCfg


# Note: 평행4절 링키지 / 서보 각도 변환은 하드웨어 전용 로직 → 실로봇의
# hardware_interface 에서 처리. Isaac Sim 은 간소화 URDF(직결 3링크) 로 동작하므로
# 정책 출력은 그대로 자연 관절각이며 URDF 의 joint limit 이 물리 한계 역할.


class VettarEnv(DirectRLEnv):
    cfg: VettarFlatEnvCfg

    def __init__(self, cfg: VettarFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2",
                "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2",
                "action_rate_l2", "feet_air_time", "feet_clearance",
                "undesired_contacts", "flat_orientation_l2",
                "joint_deviation_l2", "feet_slip",
            ]
        }

        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        # foot_F_L, foot_F_R, foot_B_L, foot_B_R — URDF 의 발끝 링크 4개.
        # calf 전체가 아니라 발끝만 "feet" 로 카운트 → 무릎 접촉은 undesired_contact 로 페널티.
        self._feet_ids, _ = self._contact_sensor.find_bodies("foot_.*")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_forces)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 화살표 마커: 파랑=명령 방향, 초록=실제 속도
        cmd_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
        cmd_cfg.prim_path = "/Visuals/Command"
        self._cmd_markers = VisualizationMarkers(cmd_cfg)
        act_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        act_cfg.prim_path = "/Visuals/Actual"
        self._act_markers = VisualizationMarkers(act_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Untrained 정책의 가우시안 노이즈가 ±3~5 쏟아내면 target 튕김 → 발광 방지.
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)
        self._update_direction_markers()

    def _update_direction_markers(self):
        """로봇 위에 화살표 2개 표시: 파랑=명령 방향, 초록=실제 속도."""
        pos = self._robot.data.root_pos_w.clone()
        pos[:, 2] += 0.25

        q = self._robot.data.root_quat_w  # wxyz
        base_yaw = torch.atan2(
            2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1 - 2 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]),
        )

        cmd_bx = self._commands[:, 0]
        cmd_by = self._commands[:, 1]
        cmd_wx = cmd_bx * torch.cos(base_yaw) - cmd_by * torch.sin(base_yaw)
        cmd_wy = cmd_bx * torch.sin(base_yaw) + cmd_by * torch.cos(base_yaw)
        cmd_yaw = torch.atan2(cmd_wy, cmd_wx)
        cmd_len = torch.sqrt(cmd_bx**2 + cmd_by**2).clamp(min=0.05).unsqueeze(-1)
        cmd_scale = torch.cat(
            [cmd_len, torch.full_like(cmd_len, 0.08), torch.full_like(cmd_len, 0.08)],
            dim=-1,
        )
        cmd_quat = torch.stack(
            [torch.cos(cmd_yaw / 2), torch.zeros_like(cmd_yaw), torch.zeros_like(cmd_yaw), torch.sin(cmd_yaw / 2)],
            dim=-1,
        )
        self._cmd_markers.visualize(pos, cmd_quat, scales=cmd_scale)

        v = self._robot.data.root_lin_vel_w
        vyaw = torch.atan2(v[:, 1], v[:, 0])
        vlen = torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2).clamp(min=0.05).unsqueeze(-1)
        v_scale = torch.cat(
            [vlen, torch.full_like(vlen, 0.08), torch.full_like(vlen, 0.08)],
            dim=-1,
        )
        v_quat = torch.stack(
            [torch.cos(vyaw / 2), torch.zeros_like(vyaw), torch.zeros_like(vyaw), torch.sin(vyaw / 2)],
            dim=-1,
        )
        self._act_markers.visualize(pos, v_quat, scales=v_scale)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._actions,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.15)

        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.10)

        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        # 0.2s 만 넘기면 양수 보상 (작은 로봇은 0.5s 체공 어려움).
        air_time = torch.sum((last_air_time - 0.2) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )

        # 발 높이 직접 보상 (0~5cm 구간 선형). 스윙 중일 때만.
        foot_z = self._robot.data.body_pos_w[:, self._feet_ids, 2]
        in_air = self._contact_sensor.data.current_contact_time[:, self._feet_ids] == 0
        target_clearance = 0.05
        feet_clearance = torch.sum(
            torch.clamp(foot_z, min=0.0, max=target_clearance) * in_air.float(),
            dim=1,
        ) * (torch.norm(self._commands[:, :2], dim=1) > 0.1)

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        current_contact_forces = torch.norm(net_contact_forces[:, 0, :], dim=-1)

        non_feet_contact_forces = current_contact_forces.clone()
        if len(self._feet_ids) > 0:
            non_feet_contact_forces[:, self._feet_ids] = 0.0

        is_contact = torch.max(non_feet_contact_forces, dim=1)[0] > 1.0
        contacts = is_contact.float()

        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # (A) 관절 정자세 유지: 현재 관절 ↔ default 차이 L2
        joint_deviation = torch.sum(
            torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos),
            dim=1,
        )

        # (B) 발 미끌림: 땅에 닿은 채 xy 평면 속도 제곱 합
        feet_vel_xy = self._robot.data.body_lin_vel_w[:, self._feet_ids, :2]
        feet_in_contact = self._contact_sensor.data.current_contact_time[:, self._feet_ids] > 0
        feet_slip = torch.sum(
            torch.sum(torch.square(feet_vel_xy), dim=-1) * feet_in_contact.float(),
            dim=-1,
        )

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "feet_clearance": feet_clearance * self.cfg.feet_clearance_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "joint_deviation_l2": joint_deviation * self.cfg.joint_deviation_reward_scale * self.step_dt,
            "feet_slip": feet_slip * self.cfg.feet_slip_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # 베이스 충돌 조기 종료 (50N 임계). ANYmal 1N 기준은 Vettar(2kg)에 너무 민감.
        if len(self._base_id) > 0:
            base_forces = torch.norm(net_contact_forces[:, 0, self._base_id], dim=-1)
            if base_forces.dim() > 1:
                base_forces = torch.max(base_forces, dim=1)[0]
            died = base_forces > 50.0
        else:
            died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # === [PLAY TEST] 특정 명령만 주고 싶을 때 True ===
        # 학습 시엔 반드시 FORCE_COMMAND = False 로 되돌릴 것.
        FORCE_COMMAND = False
        FORCED_VX, FORCED_VY, FORCED_WZ = 1.0, 0.0, 0.0
        # =============================================

        # 30% 순수 전후진 / 20% 순수 옆걸음 / 20% 순수 회전 / 30% 섞인 명령.
        n = len(env_ids)
        cmd = torch.empty(n, 3, device=self.device).uniform_(-1.0, 1.0)
        mode = torch.rand(n, device=self.device)
        pure_vx = mode < 0.30
        pure_vy = (mode >= 0.30) & (mode < 0.50)
        pure_yaw = (mode >= 0.50) & (mode < 0.70)
        cmd[pure_vx, 1] = 0.0
        cmd[pure_vx, 2] = 0.0
        cmd[pure_vy, 0] = 0.0
        cmd[pure_vy, 2] = 0.0
        cmd[pure_yaw, 0] = 0.0
        cmd[pure_yaw, 1] = 0.0

        if FORCE_COMMAND:
            cmd[:, 0] = FORCED_VX
            cmd[:, 1] = FORCED_VY
            cmd[:, 2] = FORCED_WZ

        self._commands[env_ids] = cmd

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
