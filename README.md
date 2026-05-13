# vettar — Isaac Lab 4족보행 외부 프로젝트

NVIDIA Isaac Lab 의 **External Project** 구조로 패키징된 Vettar 4족보행 로봇 RL 환경입니다.
(Isaac Lab 본체에 내장된 `isaaclab.bat --new` 템플릿 생성기에서 External 모드로 뽑은 표준 레이아웃을 따릅니다.)
Isaac Lab 본 리포지토리에 손대지 않고도 본 프로젝트 폴더 하나로 학습/재생/자산 변환을 수행할 수 있습니다.

> 원본 작업 위치: `C:\IsaacLab\source\isaaclab_*\...\vettar\` 내부 + 루트 유틸 스크립트들.
> 본 프로젝트는 그 코드를 손상 없이 외부 패키지로 옮긴 결과물입니다 (원본은 그대로 남겨둠).

---

## 디렉토리 구조

```
vettar_isaaclab/
├── README.md
├── .gitignore                       # logs/, outputs/, .venv 등 학습 산출물 제외
├── scripts/                         # 진입점 스크립트
│   ├── train.py                     # rsl_rl PPO 학습
│   ├── play.py                      # 학습 정책 재생
│   ├── smoke_test.py                # 패키지+USD 로드 빠른 검증 (~30s)
│   ├── preview_pose.py              # omni.ui 슬라이더로 관절 자세 탐색
│   ├── check_pose.py                # 물리 ON 안착 검사
│   ├── regen_usd.py                 # URDF → USD 재변환
│   ├── scale_mass.py                # URDF 총 질량 스케일
│   ├── inspect_usd.py               # USD joint limit 덤프
│   └── inspect_usd_axes.py          # USD 회전축/localRot 덤프
├── source/
│   └── vettar/                      # pip 설치 가능한 Extension 패키지
│       ├── pyproject.toml
│       ├── setup.py
│       ├── config/
│       │   └── extension.toml       # Omniverse Extension 메타데이터
│       └── vettar/                  # 실제 import 되는 Python 패키지
│           ├── __init__.py          # VETTAR_PROJECT_ROOT, VETTAR_RESOURCES_DIR 노출
│           ├── assets/
│           │   └── vettar.py        # ArticulationCfg + ImplicitActuatorCfg
│           └── tasks/direct/vettar_locomotion/
│               ├── __init__.py      # gym.register("Isaac-Vettar-Flat-Direct-v0")
│               ├── vettar_env.py
│               ├── vettar_env_cfg.py
│               └── agents/
│                   └── rsl_rl_ppo_cfg.py
└── resources/
    ├── vettar_description/          # 로봇 자산
    │   ├── urdf/vettar.urdf         # 통합 URDF (실로봇 + Isaac 공용)
    │   ├── usd/vettar.usd           # Isaac 용 변환 결과
    │   ├── usd/config.yaml          # UrdfConverter 파라미터
    │   └── meshes/*.STL             # 시각/충돌 메쉬
    └── policies/                    # 사전 학습된 정책 (외부 추론/실로봇용)
        └── vettar_flat_<timestamp>/
            ├── policy.pt            # TorchScript 추론 정책
            ├── policy.onnx          # 동일 정책의 ONNX
            ├── env.yaml             # 학습 시점의 환경 설정
            ├── agent.yaml           # PPO 하이퍼파라미터
            └── README.md            # 해당 정책 메타 정보
```

`logs/`, `outputs/` 는 `.gitignore` 로 제외되며, 학습 시점에 자동 생성됩니다.

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

# 본격 학습 (예: 3000 iter)
isaaclab -p scripts/train.py --num_envs 4096 --max_iterations 3000 --headless
```

학습 결과는 `logs/rsl_rl/vettar_flat/<timestamp>/` 에 저장 (커밋되지 않음):
- `model_*.pt` — `save_interval` (기본 50) 마다 저장되는 체크포인트
- `exported/policy.pt`, `policy.onnx` — 학습 종료 후 추론용으로 내보낸 정책
- `params/env.yaml`, `agent.yaml` — 학습 시점 설정 스냅샷
- `events.out.tfevents.*` — TensorBoard 로그

### 5) 학습된 정책 재생 (직접 학습한 경우)

```powershell
# 가장 최근 런의 가장 큰 iter 체크포인트 자동 선택 (기본 패턴 model_.*.pt)
isaaclab -p scripts/play.py --num_envs 16

# 특정 체크포인트 지정
isaaclab -p scripts/play.py --num_envs 16 --checkpoint model_2999.pt

# 특정 런 + 체크포인트 지정
isaaclab -p scripts/play.py --num_envs 16 --load_run 2026-04-24_13-24-40 --checkpoint model_2999.pt
```

> `play.py` 는 `logs/rsl_rl/vettar_flat/<load_run>/<checkpoint>` 에서 **rsl_rl 체크포인트**(`model_*.pt`)를 읽습니다.
> 동봉된 `resources/policies/.../policy.pt` 는 **TorchScript 로 내보낸 추론 전용 포맷**이라 `play.py` 가 그대로 못 읽습니다 — 다음 섹션 참고.

### 6) 동봉된 사전 학습 정책 사용

이 리포에는 한 번 학습이 끝난 정책이 함께 들어있습니다:

```
resources/policies/vettar_flat_2026-04-24_13-24-40/
├── policy.pt      # TorchScript (PyTorch 추론)
├── policy.onnx    # ONNX (C++/ROS2/엣지 런타임)
├── env.yaml       # 어떤 환경에서 학습됐는지
├── agent.yaml     # PPO 하이퍼파라미터
└── README.md      # 학습 요약
```

**용도**:
- 실로봇 / 외부 추론 노드에서 그대로 로드해 추론 (관측 차원·스케일은 `env.yaml` 과 동일해야 함).
- Isaac Lab 안의 `play.py` 로 재생하고 싶으면, 해당 폴더의 체크포인트(`model_*.pt`)가 필요하므로 본인이 다시 학습해서 `logs/` 를 채우는 것이 가장 깔끔합니다.

---

## sim2real 철학

ROS2 레이어 (`ik_node`, `gait_generator`, `imu_stabilizer`) 는 sim/real 공통.
하드웨어 전용 변환 (서보 오프셋, 평행4절 mimic) 은 실로봇의 `vettar_interface/hardware_interface.cpp` 에서만 처리.
Isaac Sim 의 정책은 자연 관절각만 출력하며, URDF joint limit 이 물리 한계 역할.

`vettar.assets.vettar` 의 `ImplicitActuatorCfg` 는 9imod DS35MG 서보 (3.44 Nm, 4.19 rad/s) 의 내부 PID 거동을 근사.
**`DCMotorCfg` 는 사용 금지** — 충격 시 각속도 상승으로 토크가 0.15 Nm 까지 떨어져 학습이 발광합니다.

---

## 라이선스

BSD-3-Clause (코드). 메쉬/URDF 는 원 저작자 표기 유지.

---

## 이름의 유래

**Vettar** 는 한국어 *"뱉어"* 처럼 발음합니다 (`/ˈbɛtʰʌ/`, "bæth-eo").

- 🐕 **뱉어** — 강아지가 뭐 주워먹으면 주인이 외치는 그 "뱉어!" — 4족보행 로봇이 개를 닮아서.
- 🅱️ **베타 (Beta)** — 알파 → 베타 → 정식 단계의 그 베타. 베타 단계라는 의미 동시에.
