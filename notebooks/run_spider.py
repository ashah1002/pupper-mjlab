"""Run the Spider notebook's steps without the viewer cells (they never finish).

In Colab, after setup and the W&B login cell:
    !cd /content/pupper-mjlab && git pull -q
    !STAGE=clips nohup python -u notebooks/run_spider.py > /content/spider_clips.log 2>&1 &
    !STAGE=walk  nohup python -u notebooks/run_spider.py > /content/spider_walk.log 2>&1 &

STAGE=clips: checks, clip library, plain replay (Q13), upload, train the clip policy, score it.
STAGE=walk:  train the spider walk with notebooks/spider_weights.json (Q14).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import wandb

from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.tricks import spider

STAGE = os.environ.get("STAGE", "clips")
ROOT = Path("/content/pupper-mjlab")
OUT = Path("/content/tricks")
OUT.mkdir(exist_ok=True)
CLIP_FILE = OUT / "spider_walk.npz"


def sh(*args):
  print("$", " ".join(map(str, args)), flush=True)
  res = subprocess.run([sys.executable, "-m", *map(str, args)], capture_output=True, text=True, cwd=ROOT)
  print(res.stdout[-5000:], res.stderr[-2000:] if res.returncode else "", flush=True)


from mjlab.scripts.train import TrainConfig, launch_training  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
import mjlab.tasks  # noqa: E402,F401  register tasks

if STAGE == "clips":
  print("=== S1 checks (cut, hold, reverse) ===", flush=True)
  assert spider.check_helpers(), "fix the FAIL lines above"
  print("\n=== S2 clip library ===", flush=True)
  library = md.concat_clips(spider.clips(), order=spider.ORDER)
  for key, value in library.report.items():
    print(f"{key:36} {value}")
  for c in library.clips:
    print(f"  clip {c['name']:12} frames {c['start']:4}-{c['end']:4} "
          f"({(c['end'] - c['start']) / 50:.1f} s) loop={c['loop']} next={c['next']}", flush=True)
  library.save_npz(CLIP_FILE)
  print("\n=== plain replay, no policy (Q13) ===", flush=True)
  sh("mjlab.tasks.tracking.config.pupper.play_motion", CLIP_FILE)
  print("\n=== upload ===", flush=True)
  sh("mjlab.tasks.tracking.config.pupper.upload_motion", CLIP_FILE, "--name", "spider_walk")
  entity = wandb.Api().default_entity
  task = "Mjlab-Tracking-Flat-Pupper-v3"
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = 4096
  agent_cfg = load_rl_cfg(task)
  agent_cfg.max_iterations = int(os.environ.get("ITERATIONS", "1000"))  # notebook: 3000
  agent_cfg.logger = "wandb"
  agent_cfg.run_name = "spider_walk_clips"
  print("\n=== train the clip policy:", agent_cfg.max_iterations, "iterations ===", flush=True)
  launch_training(task, TrainConfig(env=env_cfg, agent=agent_cfg, registry_name=f"{entity}/mjlab/spider_walk"))
  runs = sorted((ROOT / "logs/rsl_rl/pupper_tracking").glob("*spider_walk_clips*"), key=lambda p: p.stat().st_mtime)
  ckpt = max(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  print("\n=== score the clip policy:", ckpt, "===", flush=True)
  sh("mjlab.tasks.tracking.config.pupper.play_motion", CLIP_FILE, "--checkpoint", ckpt)

elif STAGE == "walk":
  weights = json.loads((ROOT / "notebooks/spider_weights.json").read_text())
  task = "Mjlab-SpiderWalk-Flat-Pupper-v3"
  env_cfg = load_env_cfg(task)
  for name, weight in weights["REWARD_WEIGHTS"].items():
    env_cfg.rewards[name].weight = weight
  env_cfg.scene.num_envs = 4096
  agent_cfg = load_rl_cfg(task)
  agent_cfg.max_iterations = int(os.environ.get("ITERATIONS", weights.get("ITERATIONS", 2000)))
  agent_cfg.logger = "wandb"
  agent_cfg.run_name = "spider_walk"
  print("=== train the spider walk:", agent_cfg.max_iterations, "iterations ===", flush=True)
  print(json.dumps(weights["REWARD_WEIGHTS"], indent=1), flush=True)
  launch_training(task, TrainConfig(env=env_cfg, agent=agent_cfg))
