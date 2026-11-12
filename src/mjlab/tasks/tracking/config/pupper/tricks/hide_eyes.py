"""Hide-and-seek (final project): Pupper sits and hides its eyes while it counts.

An interactive trick, built like sit_stay_shake:
  sit_down       stand -> sit (the worked-example sit, sit.py)
  cover          from sitting: lean right, raise the LEFT front paw up in front of
                 the face
  eyes_covered   hold the paw there (repeats while Pupper counts)
  uncover_stand  lower the paw, sit square, stand up (the exit clip)

The paw goes in front of the left side of the face, not the middle: the front-left
abduction joint can't bring the foot further in than about 5 cm left of the
body's centre line (a reachability search over targets in front of the face,
sitting at -35 degrees pitch, found nothing closer within 2 mm of IK error).
It reads as "peek-a-boo" either way; the game doesn't look while counting.

    uv run python -m mjlab.tasks.tracking.config.pupper.tricks.hide_eyes

checks the clips and writes hide_eyes.npz, the file you train on
(Mjlab-Tracking-Flat-Pupper-v3, like sit_stay_shake).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mjlab.tasks.tracking.config.pupper import motion_design as md
from mjlab.tasks.tracking.config.pupper.tricks import sit_stay_shake as sss

HERE = Path(__file__).resolve().parent
FPS = 50.0

# Get the weight over the three feet that stay down before the left paw comes up:
# shift the body right, without rolling it. shake_left rolls 5 degrees, but then
# the planted right front foot can't stay put (5.9 mm of IK error), and in plain
# physics replay a 2-degree roll with a 1 cm shift tips over ~44 degrees once the
# paw is up. A 2.5 cm shift and no roll: 0.3 mm of IK error, and replay stays up.
LEAN_DEG = 0.0
LEAN_SHIFT_M = 0.025

# The paw's target, in the body frame (x forward, y left, z up, metres): in front
# of the chest (the torso ends at x = 0.151) and as far in toward the middle as
# the left leg reaches.
PAW_AT_FACE = (0.20, 0.05, -0.02)
# On the way up: lifted off the ground first, so the paw doesn't drag.
PAW_LIFTED = (0.17, 0.07, -0.08)
COVER_S = 1.6
HOLD_S = 1.0


def reverse(motion: md.Motion) -> md.Motion:
  """The motion played backwards (your Part 6 reverse())."""
  return md.Motion(
    fps=motion.fps,
    root_pos=motion.root_pos[::-1].copy(),
    root_quat=motion.root_quat[::-1].copy(),
    joint_pos=motion.joint_pos[::-1].copy(),
  )


def join(a: md.Motion, b: md.Motion) -> md.Motion:
  """``a`` then ``b`` as one motion (b's first frame replaces a's last: they match)."""
  return md.Motion(
    fps=a.fps,
    root_pos=np.concatenate([a.root_pos[:-1], b.root_pos]),
    root_quat=np.concatenate([a.root_quat[:-1], b.root_quat]),
    joint_pos=np.concatenate([a.joint_pos[:-1], b.joint_pos]),
  )


def cover() -> md.Motion:
  """From sitting: lean right, lift the left paw, bring it up in front of the face."""
  root, rpy, feet = sss.sitting_pose()
  lean_rpy = (LEAN_DEG, rpy[1], rpy[2])  # positive roll would lift the left side
  lean_root = (root[0], root[1] - LEAN_SHIFT_M, root[2])
  R = md.quat_to_mat(md.rpy_to_quat(lean_rpy))

  def body_to_world(p):
    return tuple(float(v) for v in np.asarray(lean_root) + R @ np.asarray(p))

  def paw_at(p):
    return {**feet, "front_l": p}

  return md.build_motion(
    [
      md.Keyframe(0.0, root, rpy, feet=feet),
      md.Keyframe(0.5, lean_root, lean_rpy, feet=feet),
      md.Keyframe(1.0, lean_root, lean_rpy, feet=paw_at(body_to_world(PAW_LIFTED))),
      md.Keyframe(
        COVER_S, lean_root, lean_rpy, feet=paw_at(body_to_world(PAW_AT_FACE))
      ),
    ],
    fps=FPS,
  )


def clips() -> list[md.Clip]:
  sitting = sss.whole_sit()
  up = cover()
  stand_up = sss.cut(sitting, sss.SITTING_UNTIL_S, sitting.duration)
  return [
    md.Clip(
      "sit_down", sss.cut(sitting, 0.0, sss.SITTING_FROM_S), loop=False, next="cover"
    ),
    md.Clip("cover", up, loop=False, next="eyes_covered"),
    md.Clip(
      "eyes_covered", sss.hold(up, up.duration, HOLD_S), loop=True, next="eyes_covered"
    ),
    md.Clip("uncover_stand", join(reverse(up), stand_up), loop=False, next=None),
  ]


# Every change the robot makes, in training order. Twice through, and a few
# extra holds: Pupper stays in eyes_covered for the whole count.
ORDER: list[str] = [
  "sit_down", "cover", "eyes_covered", "eyes_covered", "eyes_covered", "uncover_stand",
  "sit_down", "cover", "eyes_covered", "eyes_covered", "uncover_stand",
]  # fmt: skip


# ----------------------------------------------------------------------- checks
def check() -> bool:
  ok = True

  def report(name, passed, detail=""):
    nonlocal ok
    ok &= passed
    print(
      f"  {'ok  ' if passed else 'FAIL'} {name}" + ("" if passed else f": {detail}")
    )

  cs = {c.name: c for c in clips()}
  up = cs["cover"].motion
  err = up.report.get("max_ik_error_m", 1.0)
  report(
    "cover: every frame reached within 2 mm (IK)", err < 0.002, f"{err * 1000:.1f} mm"
  )
  k = md.kinematics()
  paw = k.feet(up.root_pos[-1], up.root_quat[-1], up.joint_pos[-1])["front_l"]
  chest = k.body_point_world(up.root_pos[-1], up.root_quat[-1], md.CHEST)
  report(
    "cover: the paw ends up in front of the chest and at least as high",
    paw[0] > chest[0] + 0.02 and paw[2] >= chest[2],
    f"paw {np.round(paw, 3)}, chest {np.round(chest, 3)}",
  )
  for a, b in zip(ORDER[:-1], ORDER[1:], strict=True):
    gap = np.abs(cs[a].motion.joint_pos[-1] - cs[b].motion.joint_pos[0]).max()
    report(f"{a} -> {b} joins up", gap < 0.05, f"{gap:.3f} rad apart")
  checked = md.check_motion(md.concat_clips(list(cs.values()), order=ORDER))
  report(
    "nothing below the ground", checked["lowest_body_point_m"] > -0.002, str(checked)
  )
  report(
    "joint speeds under 15 rad/s", checked["max_joint_speed_rad_s"] < 15.0, str(checked)
  )
  return ok


def main() -> None:
  print("Checking the hide-eyes clips:")
  if not check():
    raise SystemExit("Fix the failing checks first.")
  library = md.concat_clips(clips(), order=ORDER)
  for key, value in library.report.items():
    print(f"{key:36} {value}")
  for c in library.clips:
    print(
      f"  clip {c['name']:14} frames {c['start']:4}-{c['end']:4} "
      f"({(c['end'] - c['start']) / FPS:.1f} s) loop={c['loop']} next={c['next']}"
    )
  print("wrote", library.save_npz(HERE / "hide_eyes.npz"))


if __name__ == "__main__":
  main()
