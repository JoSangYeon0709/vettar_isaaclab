"""resources/vettar_description/usd/vettar.usd 의 joint limit 을 직접 확인.

실행:
    isaaclab -p scripts/inspect_usd.py
"""
import math
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "vettar"))

from isaaclab.app import AppLauncher

_app_launcher = AppLauncher(headless=True)
_app = _app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402

from vettar import VETTAR_RESOURCES_DIR  # noqa: E402

USD_PATH = os.path.join(VETTAR_RESOURCES_DIR, "usd", "vettar.usd")

stage = Usd.Stage.Open(USD_PATH)

print(f"USD: {USD_PATH}\n")
print("=" * 80)
print(f"{'Joint':<40s} {'Type':<12s} {'Lower':>10s} {'Upper':>10s}")
print("=" * 80)


def fmt(v):
    if v is None:
        return "   (unset)"
    return f"{v:+.3f}°"


for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        name = prim.GetName()
        joint_type = prim.GetTypeName()
        lower = prim.GetAttribute("physics:lowerLimit")
        upper = prim.GetAttribute("physics:upperLimit")
        lower_val = lower.Get() if lower.HasValue() else None
        upper_val = upper.Get() if upper.HasValue() else None
        print(f"{name:<40s} {joint_type:<12s} {fmt(lower_val):>10s} {fmt(upper_val):>10s}")

print("\n※ USD revolute joint 의 limit 은 degree 단위로 저장됨")
print(f"※ URDF 원본 limit = ±{math.degrees(math.pi/2):.1f}° (±π/2 rad)")
print(f"※ limit 이 '(unset)' 이거나 매우 큰 값(±1e10 등)이면 물리 제한 없음")

_app.close()
