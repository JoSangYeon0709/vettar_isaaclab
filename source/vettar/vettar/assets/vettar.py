"""Configuration for the Vettar quadruped robot."""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from vettar import VETTAR_RESOURCES_DIR

##
# Configuration - Actuators.
##
# ImplicitActuatorCfg: 속도 종속 토크 감소 없는 순수 PD.
# DCMotorCfg 는 충격 시 각속도 높아지면 토크가 0.15Nm 수준까지 떨어져 학습 불안정.
VETTAR_SIMPLE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    # 실서보 (9imod DS35MG) 스펙 그대로. 35 kg·cm @ 7.4V = 3.44 Nm, 4.19 rad/s.
    # sim2real 일관성 위해 스펙 초과 금지.
    effort_limit=3.44,
    velocity_limit=4.19,
    stiffness={".*": 200.0},
    damping={".*": 5.0},
)

##
# Configuration - Articulation.
##
# USD 경로: 프로젝트 resources/ 디렉토리 기준 (어느 머신에서 clone 해도 동작).
_USD_PATH = os.path.join(VETTAR_RESOURCES_DIR, "usd", "vettar.usd").replace("\\", "/")

VETTAR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base_z 0.174: preview 에서 foot_*.z = -0.044 (@ base=0.13) 측정 → 지면 관통.
        # 0.13 + 0.044 = 0.174 로 올려서 스폰 즉시 발끝 지면 접촉.
        pos=(0.0, 0.0, 0.174),
        # 사용자가 preview 슬라이더로 확정한 자세 (2026-04-23 밤).
        # 앞다리 허벅지 ±0.5 / 뒷다리 0, 정강이: F ±1.4, B ±1.35.
        joint_pos={
            "F_L_0":  0.0,
            "F_L_1": -0.5,
            "F_L_2":  1.4,
            "F_R_3":  0.0,
            "F_R_4":  0.5,
            "F_R_5": -1.4,
            "B_L_6":  0.0,
            "B_L_7":  0.0,
            "B_L_8":  1.35,
            "B_R_9":  0.0,
            "B_R_10": 0.0,
            "B_R_11": -1.35,
        },
    ),
    actuators={"legs": VETTAR_SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
