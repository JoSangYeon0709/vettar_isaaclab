"""vettar.usd 의 revolute joint 축/변환/리미트 덤프.

URDF 의 음수축 `<axis xyz="-1 0 0">` 은 USD 변환 시 body 의 localRot 에 흡수되며
physics:axis 는 +X/+Y/+Z 중 하나로만 저장됨. 이 스크립트로 그 사상을 검증.

실행:
    isaaclab -p scripts/inspect_usd_axes.py
"""
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
print("=" * 120)
print(
    f"{'Joint':<14s} {'Axis':<5s} "
    f"{'localRot0 (w,x,y,z)':<34s} {'localRot1 (w,x,y,z)':<34s} "
    f"{'Lower':>10s} {'Upper':>10s}"
)
print("=" * 120)


def fmt_quat(q):
    if q is None:
        return "(unset)"
    imag = q.GetImaginary()
    return f"({q.GetReal():+.3f}, {imag[0]:+.3f}, {imag[1]:+.3f}, {imag[2]:+.3f})"


def fmt_limit(v):
    if v is None:
        return "(unset)"
    return f"{v:+.2f}°"


rows = []
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):
        continue
    if prim.GetTypeName() != "PhysicsRevoluteJoint":
        continue

    name = prim.GetName()
    axis = prim.GetAttribute("physics:axis").Get() if prim.GetAttribute("physics:axis").HasValue() else "?"
    lr0 = prim.GetAttribute("physics:localRot0").Get() if prim.GetAttribute("physics:localRot0").HasValue() else None
    lr1 = prim.GetAttribute("physics:localRot1").Get() if prim.GetAttribute("physics:localRot1").HasValue() else None
    lower = prim.GetAttribute("physics:lowerLimit").Get() if prim.GetAttribute("physics:lowerLimit").HasValue() else None
    upper = prim.GetAttribute("physics:upperLimit").Get() if prim.GetAttribute("physics:upperLimit").HasValue() else None

    rows.append((name, axis, lr0, lr1, lower, upper))


def sort_key(row):
    name = row[0]
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return 99


rows.sort(key=sort_key)

for name, axis, lr0, lr1, lower, upper in rows:
    print(
        f"{name:<14s} {str(axis):<5s} "
        f"{fmt_quat(lr0):<34s} {fmt_quat(lr1):<34s} "
        f"{fmt_limit(lower):>10s} {fmt_limit(upper):>10s}"
    )

print()
print("참고:")
print("  - physics:axis 는 회전축 토큰 (X/Y/Z). URDF 음수축은 localRot 로 흡수됨.")
print("  - localRot0/1 은 parent/child body 프레임에서 관절 축이 어떻게 놓이는지 결정.")

_app.close()
