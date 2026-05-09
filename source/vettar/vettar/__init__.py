"""Vettar 4족보행 로봇용 Isaac Lab 패키지.

서브모듈:
- vettar.assets : 로봇 ArticulationCfg
- vettar.tasks  : RL 환경 (Direct API)

import 시 task 들도 자동으로 gym.register() 되도록 tasks 를 임포트.
"""

import os

# 프로젝트 루트 (resources/ 가 있는 곳) — 스크립트와 cfg 가 자산을 찾을 때 사용.
# source/vettar/vettar/__init__.py 기준 → ../../../ 가 프로젝트 루트
VETTAR_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
VETTAR_RESOURCES_DIR = os.path.join(VETTAR_PROJECT_ROOT, "resources", "vettar_description")

# Task 등록 (gym.register) 트리거
from . import tasks  # noqa: F401, E402
