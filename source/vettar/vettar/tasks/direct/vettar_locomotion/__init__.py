# Copyright (c) 2025, Vettar Project
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .vettar_env import VettarEnv
from .vettar_env_cfg import VettarFlatEnvCfg

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Vettar-Flat-Direct-v0",
    entry_point="vettar.tasks.direct.vettar_locomotion:VettarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": VettarFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:VettarFlatPPORunnerCfg",
    },
)
