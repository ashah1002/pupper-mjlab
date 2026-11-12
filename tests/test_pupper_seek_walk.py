"""SeekWalk (hide-and-seek final project): the orbit commands and the seek gait."""

import numpy as np
import torch

from mjlab.tasks.pupper.mdp import sample_orbit_commands
from mjlab.tasks.pupper_gait.mdp.gait_reference import build_joint_reference_table
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import (
  SEEK_MAX_ANG_VEL_Z,
  SEEK_MAX_LIN_VEL_Y,
  SEEK_ORBIT_RADIUS,
  SEEK_ORBIT_SPEED,
)


def _orbits(n=20000, yaw_noise=0.0):
  torch.manual_seed(0)
  return sample_orbit_commands(n, SEEK_ORBIT_RADIUS, SEEK_ORBIT_SPEED, 0.1, yaw_noise)


def test_orbit_faces_the_centre():
  cmd = _orbits()
  vx, vy, wz = cmd.T
  # strafing left (vy > 0) around a point ahead means turning right (wz < 0)
  assert torch.all(vy * wz < 0)
  r = -vy / wz
  assert r.min() >= SEEK_ORBIT_RADIUS[0] - 1e-6
  assert r.max() <= SEEK_ORBIT_RADIUS[1] + 1e-6
  assert vx.abs().max() <= 0.1 + 1e-6


def test_orbit_speeds_and_both_directions():
  vy = _orbits()[:, 1]
  assert vy.abs().min() >= SEEK_ORBIT_SPEED[0] - 1e-6
  assert vy.abs().max() <= SEEK_ORBIT_SPEED[1] + 1e-6
  assert 0.45 < (vy > 0).float().mean() < 0.55


def test_orbit_turns_fast_enough_for_the_heading_hold():
  # the heading hold takes over below 0.1 rad/s and would undo the orbit's turn
  wz = _orbits()[:, 2]
  assert wz.abs().min() >= 0.1 - 1e-6


def test_orbits_fit_the_task_ranges():
  cmd = _orbits(yaw_noise=0.05)
  assert cmd[:, 1].abs().max() <= SEEK_MAX_LIN_VEL_Y
  assert cmd[:, 2].abs().max() <= SEEK_MAX_ANG_VEL_Z


def test_seek_gait_steps_in_place():
  seek = build_joint_reference_table(100, "seek")
  trot = build_joint_reference_table(100, "trot")
  assert seek.shape == (100, 12)
  assert np.all(
    np.ptp(seek, axis=0) <= np.ptp(trot, axis=0) + 1e-9
  )  # smaller than the trot
  assert np.ptp(seek, axis=0).max() > 0.3  # but the legs do step
