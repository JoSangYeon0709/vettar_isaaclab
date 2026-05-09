# vettar — Isaac Lab 4족보행 외부 프로젝트

NVIDIA Isaac Lab 의 **External Project Template** 구조로 패키징된 Vettar 4족보행 로봇 RL 환경입니다.
Isaac Lab 본 리포지토리에 손대지 않고도 본 프로젝트 폴더 하나로 학습/재생/자산 변환을 수행할 수 있습니다.

> 원본 작업 위치: `C:\IsaacLab\source\isaaclab_*\...\vettar\` 내부 + 루트 유틸 스크립트들.
> 본 프로젝트는 그 코드를 손상 없이 외부 패키지로 옮긴 결과물입니다 (원본은 그대로 남겨둠).

---

## 디렉토리 구조

```
vettar_isaaclab/
├── README.md
├── pyproject.toml (... 루트 메타는 source/vettar 가 보유)
├── .gitignore
├── scripts/                       # 진입점 스크립트
│   ├── train.py                   # rsl_rl PPO 학습
│   ├── play.py                    # 학습 정책 재생
│   ├── smoke_test.py              # 패키지+USD 로드 빠른 검증 (~30s)
│   ├── preview_pose.py            # omni.ui 슬라이더로 관절 자세 탐색
│   ├── check_pose.py              # 물리 ON 안착 검사
│   ├── regen_usd.py               # URDF → USD 재변환
│   ├── scale_mass.py              # URDF 총 질량 스케일
│   ├── inspect_usd.py             # USD joint limit 덤프
│   └── inspect_usd_axes.py        # USD 회전축/localRot 덤프
├── source/
│   └── vettar/                    # pip 설치 가능한 Extension 패키지
│       ├── pyproject.toml
│       ├── setup.py
│       ├── config/
│       │   └── extension.toml     # Omniverse Extension 메타데이터
│       └── vettar/                # 실제 import 되는 Python 패키지
│           ├── __init__.py        # VETTAR_PROJECT_ROOT, VETTAR_RESOURCES_DIR 노출
│           ├── assets/
│           │   └── vettar.py      # ArticulationCfg + ImplicitActuatorCfg
│           └── tasks/direct/vettar_locomotion/
│               ├── __init__.py    # gym.register("Isaac-Vettar-Flat-Direct-v0")
│               ├── vettar_env.py
│               ├── vettar_env_cfg.py
│               └── agents/
│                   └── rsl_rl_ppo_cfg.py
└── resources/
    └── vettar_description/
        ├── urdf/vettar.urdf       # 통합 URDF (실로봇 + Isaac 공용)
        ├── usd/vettar.usd         # Isaac 용 변환 결과
        ├── usd/config.yaml        # UrdfConverter 파라미터
        └── meshes/*.STL           # 시각/충돌 메쉬 (16MB)
```

---

## 설치

### 사전 조건
- **Isaac Sim 5.1.0** (pip 설치, `pip install isaacsim`)
- **Isaac Lab 2.3.x** (`isaaclab.bat -i` 로 본 리포 설치)
- **PyTorch 2.7.0 + CUDA 12.8** (RTX 50 시리즈는 nightly 필요)
- 가상환경 (예: `C:\isaacenv511`)

### 본 패키지 설치 (editable)

```powershell
# venv 활성화
& 'C:\isaacenv511\Scripts\Activate.ps1'

# 본 외부 프로젝트의 Python 패키지를 pip editable 로 설치
cd C:\Users\JSY\Desktop\Project\vettar_isaaclab
pip install -e .\source\vettar
```

이로써 `import vettar` 가능해지고, 자동으로 `Isaac-Vettar-Flat-Direct-v0` Gym 환경이 등록됩니다.

---

## 사용법

### 0) 설치 검증 (smoke test)

```powershell
isaaclab -p scripts/smoke_test.py
```

설치 직후 한 번 돌려서 다음을 확인:
- vettar 패키지 import 성공
- USD 경로 동적 해석 작동 (어느 머신이든)
- Gym env `Isaac-Vettar-Flat-Direct-v0` 등록
- Articulation 스폰 성공 (17 bodies, 12 joints)

`✅ ALL CHECKS PASSED` 가 떠야 다음 단계 진행.

### 1) 자세 탐색 (관절 슬라이더 UI)

```powershell
isaaclab -p scripts/preview_pose.py --shoulder 0.0 --thigh 0.4 --calf 1.35
```

- `--shoulder/--thigh/--calf` 인자로 좌우 미러 패턴 자동 적용.
- 슬라이더로 12개 관절 개별 조정. 발견한 자세 값을 `source/vettar/vettar/assets/vettar.py` 의 `init_state.joint_pos` 에 반영.

### 2) Standing 검증

```powershell
isaaclab -p scripts/check_pose.py            # GUI
isaaclab -p scripts/check_pose.py --headless # 로그만
```

`✅ Robot appears to stand stably` 가 떠야 학습 시작 가능.

### 3) URDF 질량/리미트 수정 후 USD 재생성

```powershell
python scripts/scale_mass.py --target 2.0   # 단순 XML 처리, isaaclab 불필요
isaaclab -p scripts/regen_usd.py            # USD 재변환
isaaclab -p scripts/inspect_usd.py          # 결과 검증
```

### 4) RL 학습

```powershell
# 짧게 발광 체크
isaaclab -p scripts/train.py --num_envs 16 --max_iterations 10 --headless

# 본격 학습
isaaclab -p scripts/train.py --num_envs 4096 --max_iterations 1500 --headless
```

학습 결과는 `logs/rsl_rl/vettar_flat/<timestamp>/` 에 저장.

### 5) 학습된 정책 재생

```powershell
isaaclab -p scripts/play.py --num_envs 16 --checkpoint model_1500.pt
```

---

## sim2real 철학

ROS2 레이어 (`ik_node`, `gait_generator`, `imu_stabilizer`) 는 sim/real 공통.
하드웨어 전용 변환 (서보 오프셋, 평행4절 mimic) 은 실로봇의 `vettar_interface/hardware_interface.cpp` 에서만 처리.
Isaac Sim 의 정책은 자연 관절각만 출력하며, URDF joint limit 이 물리 한계 역할.

`vettar.assets.vettar` 의 `ImplicitActuatorCfg` 는 9imod DS35MG 서보 (3.44 Nm, 4.19 rad/s) 의 내부 PID 거동을 근사.
**`DCMotorCfg` 는 사용 금지** — 충격 시 각속도 상승으로 토크가 0.15 Nm 까지 떨어져 학습이 발광합니다 (밤 세션 디버깅 결론).

---

## 원본 → 외부 프로젝트 매핑

| 원본 (C:\IsaacLab\) | 본 프로젝트 |
|---|---|
| `source/isaaclab_assets/isaaclab_assets/robots/vettar.py` | `source/vettar/vettar/assets/vettar.py` |
| `source/isaaclab_tasks/isaaclab_tasks/direct/vettar/` | `source/vettar/vettar/tasks/direct/vettar_locomotion/` |
| `resources/vettar_description/` | `resources/vettar_description/` |
| `regen_vettar_usd.py` 등 루트 유틸 | `scripts/regen_usd.py` 등 |

**Import 경로 변경:**
- `from isaaclab_assets.robots.vettar import VETTAR_CFG` → `from vettar.assets.vettar import VETTAR_CFG`
- Gym entry point: `isaaclab_tasks.direct.vettar:VettarEnv` → `vettar.tasks.direct.vettar_locomotion:VettarEnv`

**경로 하드코딩 제거:**
- `usd_path="C:/IsaacLab/resources/.../vettar.usd"` → `os.path.join(VETTAR_RESOURCES_DIR, ...)` 동적 해석.

---

## 라이선스

BSD-3-Clause (코드). 메쉬/URDF 는 원 저작자 표기 유지.

---

## 이름의 유래

**Vettar** 는 한국어 *"뱉어"* 처럼 발음합니다 (`/ˈbɛtʰʌ/`, "bæth-eo").

- 🐕 **뱉어** — 강아지가 뭐 주워먹으면 주인이 외치는 그 "뱉어!" — 4족보행 로봇이 개를 닮아서.
- 🅱️ **베타 (Beta)** — 알파 → 베타 → 정식 단계의 그 베타. 베타 단계라는 의미 동시에.
