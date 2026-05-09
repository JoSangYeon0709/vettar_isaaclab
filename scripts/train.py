"""rsl_rl 로 Vettar 정책 학습.

사용 예시:
    isaaclab -p scripts/train.py --num_envs 4096 --max_iterations 1500
    isaaclab -p scripts/train.py --num_envs 16  --max_iterations 10  --headless

기본 task = Isaac-Vettar-Flat-Direct-v0 (vettar 패키지 import 시 자동 등록).
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "vettar"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Vettar with rsl_rl.")
parser.add_argument("--task", type=str, default="Isaac-Vettar-Flat-Direct-v0", help="Gym env id.")
parser.add_argument("--num_envs", type=int, default=None, help="Override env cfg num_envs.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max iterations.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--load_run", type=str, default=".*", help="Run dir name (regex).")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "neptune"])
parser.add_argument("--log_project_name", type=str, default="vettar")
parser.add_argument("--video", action="store_true", help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-launch imports ----
import os.path as osp  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402

import vettar  # noqa: F401, E402  — gym.register 트리거
from vettar.tasks.direct.vettar_locomotion.agents import VettarFlatPPORunnerCfg  # noqa: E402


def main():
    env_cfg: DirectRLEnvCfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = VettarFlatPPORunnerCfg()
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.logger = args_cli.logger
    agent_cfg.wandb_project = args_cli.log_project_name
    agent_cfg.neptune_project = args_cli.log_project_name

    log_root = osp.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root = osp.abspath(log_root)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = osp.join(log_root, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": osp.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording training videos:")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if args_cli.resume:
        resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
        print(f"[INFO] Loading model from {resume_path}")
        runner.load(resume_path)

    dump_yaml(osp.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(osp.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(osp.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(osp.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
