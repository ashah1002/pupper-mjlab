"""How well a trained walk follows hide-and-seek's commands, in simulation.

Runs a checkpoint of Mjlab-SeekWalk-Flat-Pupper-v3 on two command sets and
prints the tracking error for each:

  orbit   every robot circles a point 0.6-1.2 m ahead while facing it
          (sideways speed + turning together), as the game does around a box
  mixed   the training mix (straight, turn in place, stand, random, orbits)

    python -u notebooks/eval_seek_walk.py logs/rsl_rl/pupper_seek_walk/<run>/model_2999.pt

The numbers: mean |error| of sideways speed (m/s), forward speed (m/s) and turn
rate (rad/s), measured on the robot's body velocity against the command it was
given; the orbit's radius error (commanded vs walked, where both turn); and how
many robots fell.
"""

from __future__ import annotations

import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  register tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.pupper.mdp import OrbitVelocityCommandCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-SeekWalk-Flat-Pupper-v3"
NUM_ENVS = 512
SECONDS = 12.0
SETTLE_S = 2.0  # skip the first seconds after each reset / command change


def evaluate(checkpoint: str, mode: str, device: str) -> dict:
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  twist = env_cfg.commands["twist"]
  assert isinstance(twist, OrbitVelocityCommandCfg)
  twist.resampling_time_range = (SECONDS + 1.0, SECONDS + 1.0)  # one command per run
  if mode == "orbit":
    twist.rel_orbit_envs = 1.0
    twist.rel_standing_envs = 0.0
    twist.rel_turn_envs = 0.0
    twist.rel_forward_envs = 0.0
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
  agent_cfg = load_rl_cfg(TASK)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    env, asdict(agent_cfg), device=device
  )
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  base = env.unwrapped
  term = base.command_manager.get_term("twist")
  assert term is not None
  robot = base.scene["robot"]
  obs = env.get_observations()
  steps = int(SECONDS / base.step_dt)
  settle = int(SETTLE_S / base.step_dt)
  err = torch.zeros(3, device=device)
  r_err, r_n, n = 0.0, 0, 0
  fell = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
  for k in range(steps):
    with torch.no_grad():
      obs, _, dones, _ = env.step(policy(obs))
    fell |= dones.bool() & (base.episode_length_buf > 0)
    if k < settle:
      continue
    cmd = term.command  # what the policy was asked for (with the heading hold)
    v = torch.stack(
      [
        robot.data.root_link_lin_vel_b[:, 0],
        robot.data.root_link_lin_vel_b[:, 1],
        robot.data.root_link_ang_vel_b[:, 2],
      ],
      dim=-1,
    )
    ok = ~fell
    err += (v[ok] - cmd[ok]).abs().sum(0)
    n += int(ok.sum())
    turning = (
      ok & (cmd[:, 2].abs() > 0.1) & (v[:, 2].abs() > 0.05) & (cmd[:, 1].abs() > 0.05)
    )
    if turning.any():
      r_cmd = -cmd[turning, 1] / cmd[turning, 2]
      r_walk = -v[turning, 1] / v[turning, 2]
      r_err += float((r_walk - r_cmd).abs().clamp(max=2.0).sum())
      r_n += int(turning.sum())
  e = (err / max(n, 1)).tolist()
  return {
    "vx_err": e[0],
    "vy_err": e[1],
    "wz_err": e[2],
    "radius_err_m": r_err / max(r_n, 1) if mode == "orbit" else float("nan"),
    "fell": float(fell.float().mean()),
  }


def main() -> None:
  if len(sys.argv) != 2:
    raise SystemExit(__doc__)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  head = ("commands", "|vx err|", "|vy err|", "|wz err|", "radius err", "fell")
  print(
    f"{head[0]:8} {head[1]:>9} {head[2]:>9} {head[3]:>9} {head[4]:>11} {head[5]:>6}"
  )
  for mode in ("orbit", "mixed"):
    r = evaluate(sys.argv[1], mode, device)
    print(
      f"{mode:8} {r['vx_err']:9.3f} {r['vy_err']:9.3f} {r['wz_err']:9.3f} "
      f"{r['radius_err_m']:11.3f} {r['fell']:6.1%}",
      flush=True,
    )


if __name__ == "__main__":
  main()
