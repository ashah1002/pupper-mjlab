"""Section 4 + 7 of the Sit Stay Shake notebook in one go, for Colab:
plain replay (no policy) vs the trained policy, then a shareable 3D viewer of the policy.

    %run /content/pupper-mjlab/notebooks/score_sit_stay_shake.py
"""

import re
import subprocess
import sys
import time
from pathlib import Path

MOTION_FILE = Path("/content/tricks/sit_stay_shake.npz")
PLAY = [sys.executable, "-m", "mjlab.tasks.tracking.config.pupper.play_motion", str(MOTION_FILE)]


def run(cmd, title):
  print(f"\n=== {title} ===", flush=True)
  res = subprocess.run(cmd, capture_output=True, text=True, cwd="/content/pupper-mjlab")
  print(res.stdout[-4000:], (res.stderr[-2000:] if res.returncode else ""), flush=True)


runs = sorted(Path("/content/pupper-mjlab/logs/rsl_rl/pupper_tracking").glob("*"),
              key=lambda p: p.stat().st_mtime)
assert runs, "No training runs found."
ckpt = max(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
print("checkpoint:", ckpt, flush=True)

run(PLAY, "plain replay, no policy (section 4)")
run(PLAY + ["--checkpoint", str(ckpt)], "trained policy (section 7)")

print("\n=== 3D viewer of the policy (public link) ===", flush=True)
log = Path("/content/viewer.log")
subprocess.Popen(
  PLAY + ["--checkpoint", str(ckpt), "--viewer", "viser", "--share", "True", "--loops", "3"],
  stdout=log.open("w"), stderr=subprocess.STDOUT, cwd="/content/pupper-mjlab", start_new_session=True,
)
for _ in range(120):
  time.sleep(2)
  m = re.search(r"Open the viewer:\s*(\S+)", log.read_text(errors="ignore"))
  if m:
    print("VIEWER:", m.group(1), flush=True)
    break
else:
  print("no share link yet; see /content/viewer.log", flush=True)
