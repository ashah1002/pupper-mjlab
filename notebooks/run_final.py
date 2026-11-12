"""Train the hide-and-seek final project's two policies (Colab, like run_spider.py).

In Colab, after setup and the W&B login cell (the setup clones ma-xr/pupper-mjlab;
this branch lives on the fork ashah1002/pupper-mjlab):
    !cd /content/pupper-mjlab && git remote add seek https://github.com/ashah1002/pupper-mjlab.git
    !cd /content/pupper-mjlab && git fetch -q seek hide-and-seek && git checkout -q -b hide-and-seek seek/hide-and-seek
    !STAGE=hide_eyes nohup python -u notebooks/run_final.py > /content/hide_eyes.log 2>&1 &
    !STAGE=seek_walk nohup python -u notebooks/run_final.py > /content/seek_walk.log 2>&1 &

STAGE=hide_eyes: check the clips, build hide_eyes.npz, plain replay, upload,
                 train the tracking policy (Mjlab-Tracking-Flat-Pupper-v3), score it.
STAGE=seek_walk: train Mjlab-SeekWalk-Flat-Pupper-v3 with notebooks/seek_walk_weights.json.

Then on the robot (hide-and-seek-pupper repo):
    python3 robot/get_trick.py <entity>/mjlab/<hide_eyes run> --name hide_eyes
    python3 scripts/get_seek_walk.py <entity>/mjlab/<seek_walk run>
and a TA's OK before either runs on the robot for the first time.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import wandb

from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.tricks import hide_eyes

STAGE = os.environ.get("STAGE", "hide_eyes")
ROOT = Path(os.environ.get("REPO", "/content/pupper-mjlab"))
OUT = Path(os.environ.get("OUT", "/content/tricks"))
OUT.mkdir(exist_ok=True)
CLIP_FILE = OUT / "hide_eyes.npz"


def sh(*args):
  print("$", " ".join(map(str, args)), flush=True)
  res = subprocess.run(
    [sys.executable, "-m", *map(str, args)], capture_output=True, text=True, cwd=ROOT
  )
  print(res.stdout[-5000:], res.stderr[-2000:] if res.returncode else "", flush=True)


import mjlab.tasks  # noqa: E402,F401  register tasks
from mjlab.scripts.train import TrainConfig, launch_training  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

if STAGE == "hide_eyes":
  print("=== checks ===", flush=True)
  assert hide_eyes.check(), "fix the FAIL lines above"
  library = md.concat_clips(hide_eyes.clips(), order=hide_eyes.ORDER)
  for c in library.clips:
    print(
      f"  clip {c['name']:14} frames {c['start']:4}-{c['end']:4} loop={c['loop']} "
      f"next={c['next']}",
      flush=True,
    )
  library.save_npz(CLIP_FILE)
  print("\n=== plain replay, no policy ===", flush=True)
  sh("mjlab.tasks.tracking.config.pupper.play_motion", CLIP_FILE)
  print("\n=== upload ===", flush=True)
  sh(
    "mjlab.tasks.tracking.config.pupper.upload_motion", CLIP_FILE, "--name", "hide_eyes"
  )
  entity = wandb.Api().default_entity
  task = "Mjlab-Tracking-Flat-Pupper-v3"
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = 4096
  agent_cfg = load_rl_cfg(task)
  agent_cfg.max_iterations = int(os.environ.get("ITERATIONS", "3000"))
  agent_cfg.logger = "wandb"
  agent_cfg.run_name = "hide_eyes"
  print("\n=== train:", agent_cfg.max_iterations, "iterations ===", flush=True)
  launch_training(
    task,
    TrainConfig(
      env=env_cfg, agent=agent_cfg, registry_name=f"{entity}/mjlab/hide_eyes"
    ),
  )
  runs = sorted(
    (ROOT / "logs/rsl_rl/pupper_tracking").glob("*hide_eyes*"),
    key=lambda p: p.stat().st_mtime,
  )
  ckpt = max(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  print("\n=== score the policy:", ckpt, "===", flush=True)
  sh("mjlab.tasks.tracking.config.pupper.play_motion", CLIP_FILE, "--checkpoint", ckpt)

elif STAGE == "seek_walk":
  weights = json.loads((ROOT / "notebooks/seek_walk_weights.json").read_text())
  task = "Mjlab-SeekWalk-Flat-Pupper-v3"
  env_cfg = load_env_cfg(task)
  for name, weight in weights["REWARD_WEIGHTS"].items():
    env_cfg.rewards[name].weight = weight
  env_cfg.scene.num_envs = 4096
  agent_cfg = load_rl_cfg(task)
  agent_cfg.max_iterations = int(
    os.environ.get("ITERATIONS", weights.get("ITERATIONS", 2000))
  )
  agent_cfg.logger = "wandb"
  agent_cfg.run_name = "seek_walk"
  print(
    "=== train the seek walk:", agent_cfg.max_iterations, "iterations ===", flush=True
  )
  print(json.dumps(weights["REWARD_WEIGHTS"], indent=1), flush=True)
  launch_training(task, TrainConfig(env=env_cfg, agent=agent_cfg))

else:
  raise SystemExit(f"STAGE must be hide_eyes or seek_walk, not {STAGE!r}")
