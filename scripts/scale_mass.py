"""Vettar URDF 의 모든 링크 mass / inertia 를 TARGET_TOTAL_MASS 에 맞춰 scale.

현재 상태 무관하게 목표 총 질량으로 맞춰줌 — 반복 실행해도 누적 안 됨.

사용법:
    python scripts/scale_mass.py [--target 2.0]

※ 단순 XML 처리이므로 isaaclab.bat 필요 없음.
※ 실행 후 scripts/regen_usd.py 돌려서 USD 재생성 필요.
"""
import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

# 패키지 경로 등록
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "vettar"))
from vettar import VETTAR_RESOURCES_DIR  # noqa: E402

URDF_PATH = os.path.join(VETTAR_RESOURCES_DIR, "urdf", "vettar.urdf")
BACKUP_PATH = URDF_PATH + ".bak"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--target",
    type=float,
    default=2.0,
    help="목표 총 질량 (kg). 실로봇 컴포넌트 합산 기준.",
)
args = parser.parse_args()

shutil.copy(URDF_PATH, BACKUP_PATH)
print(f"[backup] {BACKUP_PATH}")

tree = ET.parse(URDF_PATH)
root = tree.getroot()

current_total = 0.0
for link in root.iter("link"):
    inertial = link.find("inertial")
    if inertial is None:
        continue
    mass_elem = inertial.find("mass")
    if mass_elem is not None:
        current_total += float(mass_elem.get("value"))

if current_total == 0:
    raise RuntimeError("URDF 에 mass 가 0 이거나 찾을 수 없음")

scale = args.target / current_total
print(f"[current] total mass = {current_total:.4f} kg → target {args.target} kg → scale x{scale:.4f}")

total_after = 0.0
n_links = 0
for link in root.iter("link"):
    inertial = link.find("inertial")
    if inertial is None:
        continue
    mass_elem = inertial.find("mass")
    inertia_elem = inertial.find("inertia")

    if mass_elem is not None:
        new = float(mass_elem.get("value")) * scale
        mass_elem.set("value", f"{new:.15g}")
        total_after += new

    if inertia_elem is not None:
        for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            if key in inertia_elem.attrib:
                val = float(inertia_elem.get(key))
                inertia_elem.set(key, f"{val * scale:.15g}")

    n_links += 1

tree.write(URDF_PATH, xml_declaration=True, encoding="utf-8")
print(f"[scaled] {n_links} links")
print(f"  total mass: {current_total:.4f} kg  →  {total_after:.4f} kg  (target {args.target} kg)")
print(f"\n다음 단계: isaaclab -p scripts/regen_usd.py  (USD 재생성)")
