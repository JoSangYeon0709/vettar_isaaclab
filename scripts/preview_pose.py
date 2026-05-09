"""Vettar 로봇을 지정 관절각으로 프리뷰 — 중력 OFF, 슬라이더 UI 로 관절 탐색.

발견한 자세 값을 source/vettar/vettar/assets/vettar.py 의 init_state.joint_pos 에 반영.

실행:
    isaaclab -p scripts/preview_pose.py
    isaaclab -p scripts/preview_pose.py --shoulder 0.16 --thigh 0.0 --calf 1.18
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "vettar"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--shoulder", type=float, default=0.0,
                    help="어깨 각도 (rad). 대각선 미러 적용: FL+BR 양수, FR+BL 음수")
parser.add_argument("--thigh", type=float, default=0.0,
                    help="허벅지 각도 (rad). 좌우 미러 적용: L 양수, R 음수")
parser.add_argument("--calf", type=float, default=0.785,
                    help="정강이 각도 (rad). 좌우 미러 적용: L 양수, R 음수")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from vettar.assets.vettar import VETTAR_CFG  # noqa: E402


_sh = args.shoulder
_th = args.thigh
_ca = args.calf

JOINT_POS = {
    "F_L_0": +_sh,  "F_R_3": -_sh,  "B_L_6": -_sh,  "B_R_9": +_sh,
    "F_L_1": +_th,  "F_R_4": -_th,  "B_L_7": +_th,  "B_R_10": -_th,
    "F_L_2": +_ca,  "F_R_5": -_ca,  "B_L_8": +_ca,  "B_R_11": -_ca,
}
BASE_Z = 0.13


def main():
    # device="cpu": GPU direct API 끔. gravity=0: 공중 부양.
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 100, device="cpu", gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.7, 0.7, 0.5], [0.0, 0.0, 0.2])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )

    robot_cfg = VETTAR_CFG.replace(
        prim_path="/World/Robot",
        init_state=VETTAR_CFG.init_state.replace(
            pos=(0.0, 0.0, BASE_Z),
            joint_pos=JOINT_POS,
        ),
    )
    robot = Articulation(robot_cfg)

    sim.reset()

    device = robot.device
    joint_values = torch.tensor(
        [JOINT_POS[robot.joint_names[i]] for i in range(len(robot.joint_names))],
        device=device,
    ).unsqueeze(0)
    zeros = torch.zeros_like(joint_values)
    robot.write_joint_state_to_sim(joint_values, zeros)
    robot.write_data_to_sim()

    print("\n=== Joint positions (as commanded) ===")
    for name in sorted(JOINT_POS.keys()):
        print(f"  {name:<8s} = {JOINT_POS[name]:+.4f} rad")

    sim.render()
    print("\n=== Body-frame key positions ===")
    print(f"  base_link z = {BASE_Z:.3f} m  (fixed for preview)")
    for i, name in enumerate(robot.body_names):
        pos = robot.data.body_pos_w[0, i].cpu().tolist()
        print(f"  {name:<20s}  world=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")

    # ===== 슬라이더 UI =====
    import omni.ui as ui

    JOINT_LIMITS = {
        "F_L_0": (-1.5708, 1.5708),   "F_R_3": (-1.5708, 1.5708),
        "B_L_6": (-1.5708, 1.5708),   "B_R_9": (-1.5708, 1.5708),
        "F_L_1": (-1.5708, 1.5708),   "F_R_4": (-1.5708, 1.5708),
        "B_L_7": (-1.5708, 1.5708),   "B_R_10": (-1.5708, 1.5708),
        # 정강이는 좌/우 비대칭
        "F_L_2": (-0.13, 1.57),   "F_R_5": (-1.57, 0.13),
        "B_L_8": (-0.13, 1.57),   "B_R_11": (-1.57, 0.13),
    }
    joint_order = list(JOINT_LIMITS.keys())

    slider_models = {}
    window = ui.Window("Vettar Joint Control", width=460, height=560)
    with window.frame:
        with ui.VStack(spacing=4):
            ui.Label("각 관절 슬라이더 (URDF 리미트 반영)", height=22)
            ui.Label("베이스 공중 고정 · 중력 OFF · 매 프레임 상태 덮어씀", height=22)
            ui.Spacer(height=6)
            for name in joint_order:
                lo, hi = JOINT_LIMITS[name]
                with ui.HStack(height=22):
                    ui.Label(name, width=50)
                    ui.Label(f"[{lo:+.2f}, {hi:+.2f}]", width=110)
                    s = ui.FloatSlider(min=lo, max=hi, step=0.01, precision=3)
                    init_val = max(lo, min(hi, JOINT_POS[name]))
                    s.model.set_value(init_val)
                    slider_models[name] = s.model

    root_pose = torch.tensor(
        [[0.0, 0.0, BASE_Z, 1.0, 0.0, 0.0, 0.0]],
        device=device,
    )
    root_vel = torch.zeros((1, 6), device=device)

    sim_joint_order = list(robot.joint_names)
    missing = set(joint_order) - set(sim_joint_order)
    if missing:
        print(f"[WARN] sim 에 없는 관절 이름: {missing}")

    print("\n=== 준비 완료 ===")
    print(f"sim joint order: {sim_joint_order}")
    print("창 'Vettar Joint Control' 에서 슬라이더 움직이면 관절 단위 테스트 가능.\n")

    step_counter = 0
    while simulation_app.is_running():
        vals_list = [slider_models[name].as_float for name in sim_joint_order]
        vals = torch.tensor([vals_list], device=device)
        zero_vel = torch.zeros_like(vals)

        robot.write_joint_state_to_sim(vals, zero_vel)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(root_vel)

        sim.step(render=True)

        step_counter += 1
        if step_counter % 120 == 0:
            print("현재 값: " + " ".join(
                f"{n}={slider_models[n].as_float:+.2f}" for n in joint_order
            ))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
