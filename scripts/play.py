"""학습된 Vettar 정책 재생 (Xbox 패드 명령은 별도 노드/스크립트로 입힐 것).

사용 예시:
    isaaclab -p scripts/play.py --num_envs 16
    isaaclab -p scripts/play.py --num_envs 16 --checkpoint model_1500.pt
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "source", "vettar"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained Vettar policy.")
parser.add_argument("--task", type=str, default="Isaac-Vettar-Flat-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--load_run", type=str, default=".*", help="Run dir (regex).")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt", help="Checkpoint filename pattern.")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os.path as osp  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402

import vettar  # noqa: F401, E402  — gym.register 트리거
from vettar.tasks.direct.vettar_locomotion.agents import VettarFlatPPORunnerCfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = VettarFlatPPORunnerCfg()

    log_root = osp.abspath(osp.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
    print(f"[INFO] Loading model from {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=osp.join(osp.dirname(resume_path), "videos", "play"),
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
