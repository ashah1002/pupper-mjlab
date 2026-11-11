"""Run the Sit Stay Shake notebook's steps in order, without the viewer cells (which never finish).

In Colab, after setup and the W&B login cell:
    !cd /content/pupper-mjlab && git pull -q
    %run /content/pupper-mjlab/notebooks/run_sit_stay_shake.py
"""

import os
import subprocess
import sys
from pathlib import Path

import wandb

SKIP_REPLAY = os.environ.get("SKIP_REPLAY") == "1"

from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.tricks import sit_stay_shake as t

TRICK_NAME = "sit_stay_shake"
OUT = Path("/content/tricks")
OUT.mkdir(exist_ok=True)


def sh(*args):
  print("$", " ".join(map(str, args)), flush=True)
  # Colab doesn't show a subprocess's own output: capture it and print it here.
  res = subprocess.run(
    [sys.executable, "-m", *map(str, args)], capture_output=True, text=True, check=False
  )
  print(res.stdout[-6000:], res.stderr[-3000:] if res.returncode else "", flush=True)


print("=== 4. checks (cut, hold, mirror) ===")
assert t.check_helpers(), "fix the FAIL lines above"

print("\n=== 4. clip library ===")
library = md.concat_clips(t.clips(), order=t.ORDER)
for key, value in library.report.items():
  print(f"{key:36} {value}")
for c in library.clips:
  print(f"  clip {c['name']:12} frames {c['start']:4}-{c['end']:4} "
        f"({(c['end'] - c['start']) / 50:.1f} s) loop={c['loop']} next={c['next']}")
MOTION_FILE = library.save_npz(OUT / f"{TRICK_NAME}.npz")
print("saved", MOTION_FILE)

print("\n=== 4. replay with physics, no policy (Q11) ===")
if not SKIP_REPLAY:
  sh("mjlab.tasks.tracking.config.pupper.play_motion", MOTION_FILE)

print("\n=== 5. upload the motion to W&B ===")
sh("mjlab.tasks.tracking.config.pupper.upload_motion", MOTION_FILE, "--name", TRICK_NAME)
MOTION_ARTIFACT = f"{wandb.Api().default_entity}/mjlab/{TRICK_NAME}"
print("motion artifact:", MOTION_ARTIFACT)

print("\n=== 6. train ===")
from mjlab.scripts.train import TrainConfig, launch_training  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
import mjlab.tasks  # noqa: E402,F401  register tasks

TASK = "Mjlab-Tracking-Flat-Pupper-v3"
env_cfg = load_env_cfg(TASK)
env_cfg.scene.num_envs = 4096
agent_cfg = load_rl_cfg(TASK)
agent_cfg.max_iterations = int(os.environ.get("ITERATIONS", "1000"))  # handout: 3000
print("iterations:", agent_cfg.max_iterations, flush=True)
agent_cfg.logger = "wandb"
agent_cfg.run_name = TRICK_NAME
launch_training(TASK, TrainConfig(env=env_cfg, agent=agent_cfg, registry_name=MOTION_ARTIFACT))
