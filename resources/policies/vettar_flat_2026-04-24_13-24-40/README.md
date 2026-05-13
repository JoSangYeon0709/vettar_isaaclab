# vettar_flat 정책 — 2026-04-24_13-24-40

`Isaac-Vettar-Flat-Direct-v0` 환경에서 rsl_rl PPO 로 학습된 vettar 4족보행 정책.

## 학습 요약

| 항목 | 값 |
|---|---|
| 환경 | `Isaac-Vettar-Flat-Direct-v0` |
| 알고리즘 | PPO (rsl_rl) |
| 정책 네트워크 | MLP |
| 학습 iteration | 3000 (`model_0` ~ `model_2999`) |
| `num_steps_per_env` | 24 |
| seed | 42 |
| learning rate | 1e-3 |
| save interval | 50 iter |
| 원본 로그 경로 | `logs/rsl_rl/vettar_flat/2026-04-24_13-24-40/` (로컬, 깃 추적 제외) |

## 파일

- `policy.pt` — 추론용 TorchScript 정책 (PyTorch).
- `policy.onnx` — 동일 정책의 ONNX 포맷 (C++/ROS2/엣지 런타임용).
- `env.yaml` — 학습 시점의 환경 설정 (관측/액션/보상/도메인 랜덤화 등).
- `agent.yaml` — PPO 하이퍼파라미터 및 네트워크 구조.

## 사용 (Isaac Lab `play.py` 기준 예시)

```powershell
# 원본 체크포인트 디렉토리 구조(로컬)로 재생
isaaclab -p scripts/play.py --num_envs 16 --checkpoint model_2999.pt

# exported policy.pt 만 가지고 추론하려면 별도 추론 스크립트 필요
#   (관측 차원/스케일은 env.yaml 과 동일해야 함)
```

## 주의

- 정책은 `env.yaml` 의 관측 정의(차원·순서·스케일)에 묶여 있습니다.
  환경 설정이 바뀐 상태에서 그대로 추론하면 발광합니다.
- 액추에이터는 `ImplicitActuatorCfg` (9imod DS35MG 근사). `DCMotorCfg` 는 사용 금지.
