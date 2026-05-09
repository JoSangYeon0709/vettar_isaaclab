"""resources/vettar_description/urdf/vettar.urdf 를 읽어 vettar.usd 를 재생성.

실행:
    isaaclab -p scripts/regen_usd.py
또는 (Isaac Lab 가 PATH 에 없을 때):
    & 'C:/IsaacLab/isaaclab.bat' -p scripts/regen_usd.py
"""
import os
import sys

# 패키지 import 가능하도록 source/vettar 추가
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "vettar"))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from vettar import VETTAR_RESOURCES_DIR  # noqa: E402


URDF_PATH = os.path.join(VETTAR_RESOURCES_DIR, "urdf", "vettar.urdf")
USD_DIR = os.path.join(VETTAR_RESOURCES_DIR, "usd")
USD_NAME = "vettar.usd"

cfg = UrdfConverterCfg(
    asset_path=URDF_PATH,
    usd_dir=USD_DIR,
    usd_file_name=USD_NAME,
    force_usd_conversion=True,
    make_instanceable=True,
    fix_base=False,
    merge_fixed_joints=False,
    convert_mimic_joints_to_normal_joints=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
    ),
    collider_type="convex_hull",
    self_collision=False,
    replace_cylinders_with_capsules=False,
    collision_from_visuals=False,
)

print(f"[regen_usd] Converting {URDF_PATH} ...")
converter = UrdfConverter(cfg)
print(f"[regen_usd] Done. USD written to: {converter.usd_path}")

simulation_app.close()
