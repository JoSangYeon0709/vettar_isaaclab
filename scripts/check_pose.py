"""Vettar 로봇을 기본 관절 위치로 스폰해서 물리 안착 후 standing 자세 검사.

실행:
    isaaclab -p scripts/check_pose.py            # GUI
    isaaclab -p scripts/check_pose.py --headless # 로그만
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "vettar"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check Vettar standing pose.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from vettar.assets.vettar import VETTAR_CFG  # noqa: E402


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 200)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.6, 0.4], [0.0, 0.0, 0.15])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85))
    )

    robot_cfg = VETTAR_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()

    print("\n=== Vettar standing-pose check ===")
    print(f"Joint names ({len(robot.joint_names)}): {robot.joint_names}")
    print(f"Default joint positions (rad):")
    for name, value in zip(robot.joint_names, robot.data.default_joint_pos[0].cpu().tolist()):
        print(f"  {name:<8s} = {value:+.4f}")
    print(f"Initial base height: {robot.data.root_pos_w[0, 2].item():.3f} m")

    settle_steps = 1000
    print(f"\nHolding default pose for {settle_steps * sim.get_physics_dt():.1f}s ...")
    for i in range(settle_steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

        if i % 100 == 0:
            base_z = robot.data.root_pos_w[0, 2].item()
            g_b = robot.data.projected_gravity_b[0].cpu()
            print(f"  t={i*sim.get_physics_dt():4.2f}s  base_z={base_z:.3f}m  "
                  f"gravity_b=[{g_b[0]:+.2f}, {g_b[1]:+.2f}, {g_b[2]:+.2f}]")

    base_z = robot.data.root_pos_w[0, 2].item()
    g_b = robot.data.projected_gravity_b[0].cpu()
    jp = robot.data.joint_pos[0].cpu()

    print("\n=== After settle ===")
    print(f"Base height: {base_z:.3f} m  (target ~0.12 ~ 0.18 m)")
    print(f"Gravity in body frame: [{g_b[0]:+.3f}, {g_b[1]:+.3f}, {g_b[2]:+.3f}]  (standing: ~[0, 0, -1])")
    print("Joint positions after settle:")
    for name, value in zip(robot.joint_names, jp.tolist()):
        print(f"  {name:<8s} = {value:+.4f}")

    verdict = []
    if base_z < 0.10:
        verdict.append("❌ Robot collapsed (base too low, <10cm)")
    elif base_z > 0.25:
        verdict.append("⚠️ Robot base higher than expected (leg overextended?)")
    if abs(g_b[0]) > 0.3 or abs(g_b[1]) > 0.3:
        verdict.append("⚠️ Robot tilted")
    near_limit = (jp.abs() > 1.50).sum().item()
    if near_limit >= 2:
        verdict.append(f"❌ {near_limit} joints pinned at limit (sim instability)")
    if not verdict:
        verdict.append("✅ Robot appears to stand stably")
    print("\nVerdict:")
    for v in verdict:
        print(f"  {v}")

    if not args.headless:
        print("\n(GUI 닫으면 종료)")
        while simulation_app.is_running():
            sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
