"""미니 smoke test: vettar 패키지 + USD + ArticulationCfg 가 제대로 로드되는지만 확인.

settle 루프 없이 spawn 직후 종료. 빠르게 (10~30s) 결과 확인 가능.

실행:
    isaaclab -p scripts/smoke_test.py
"""
import os
import sys

# unbuffered + utf-8 stdout (Windows cp949 콘솔에서 한글/이모지 print 실패 회피)
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "vettar"))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

print("[smoke] AppLauncher booted", flush=True)

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

print("[smoke] importing vettar package...", flush=True)
from vettar.assets.vettar import VETTAR_CFG  # noqa: E402
import vettar  # noqa: F401, E402  → tasks.direct.vettar_locomotion 도 같이 등록

print(f"[smoke] VETTAR_RESOURCES_DIR = {vettar.VETTAR_RESOURCES_DIR}", flush=True)

import gymnasium as gym  # noqa: E402

env_ids = [s for s in gym.registry.keys() if "Vettar" in s]
print(f"[smoke] Registered Vettar envs: {env_ids}", flush=True)

sim_cfg = sim_utils.SimulationCfg(dt=1 / 200)
sim = sim_utils.SimulationContext(sim_cfg)
print("[smoke] SimulationContext created", flush=True)

sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

robot = Articulation(VETTAR_CFG.replace(prim_path="/World/Robot"))
print("[smoke] Articulation created", flush=True)

sim.reset()
print(f"[smoke] joint_names={robot.joint_names}", flush=True)
print(f"[smoke] body_names={robot.body_names}", flush=True)
print(f"[smoke] num_joints={len(robot.joint_names)}", flush=True)
print(f"[smoke] num_bodies={len(robot.body_names)}", flush=True)
print(f"[smoke] base_z={robot.data.root_pos_w[0, 2].item():.3f}m", flush=True)

assert len(robot.joint_names) == 12, f"expected 12 joints, got {len(robot.joint_names)}"
assert len(robot.body_names) == 17, f"expected 17 bodies, got {len(robot.body_names)}"
assert "Isaac-Vettar-Flat-Direct-v0" in env_ids, "gym env not registered"

print("[smoke] ✅ ALL CHECKS PASSED — vettar 외부 프로젝트 정상 동작", flush=True)

simulation_app.close()
