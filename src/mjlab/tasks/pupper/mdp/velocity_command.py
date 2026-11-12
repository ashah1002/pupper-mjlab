"""Pupper velocity commands: an added pure-turn mode, and orbit commands.

Extends mjlab's ``UniformVelocityCommand`` (which already provides forward-only
"walk straight" commands via ``rel_forward_envs``) with ``rel_turn_envs``: a
fraction of environments that get a pure yaw command (zero linear velocity, a
non-trivial turn rate), so the policy sees plenty of turn-in-place and
straight-line examples rather than only random x/y/yaw mixtures.

``OrbitVelocityCommand`` (the hide-and-seek final project's SeekWalk task) adds
``rel_orbit_envs``: environments that walk sideways around a point straight
ahead while facing it -- vy and wz coupled as wz = -vy / r -- the motion Pupper
makes walking around a box to look into it. Uniform commands rarely produce
that coupling, so a policy trained only on them tracks it loosely.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class PupperVelocityCommand(UniformVelocityCommand):
  cfg: "PupperVelocityCommandCfg"  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: "PupperVelocityCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.is_turn_env = torch.zeros_like(self.is_standing_env)
    # Heading (yaw) captured each time a command is (re)sampled, so a no-yaw
    # command can be rewarded for holding this heading (anti-drift). Captured
    # lazily on the next update, when heading_w is fresh (at resample time -- e.g.
    # during reset -- the root pose may not yet be reflected in the sim data).
    self.heading_hold_target = torch.zeros(self.num_envs, device=self.device)
    self._capture_heading = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _update_command(self) -> None:
    super()._update_command()
    if self._capture_heading.any():
      ids = self._capture_heading.nonzero(as_tuple=False).flatten()
      self.heading_hold_target[ids] = self.robot.data.heading_w[ids]
      self._capture_heading[ids] = False

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._capture_heading[env_ids] = True  # recapture heading on the next update
    r = torch.empty(len(env_ids), device=self.device)
    self.is_turn_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_turn_envs
    turn_ids = env_ids[self.is_turn_env[env_ids]]
    if len(turn_ids) > 0:
      # Pure yaw: zero linear, keep a meaningful turn magnitude (sign preserved).
      yaw = self.vel_command_b[turn_ids, 2]
      sign = torch.where(yaw >= 0, 1.0, -1.0)
      self.vel_command_b[turn_ids, 0] = 0.0
      self.vel_command_b[turn_ids, 1] = 0.0
      self.vel_command_b[turn_ids, 2] = sign * yaw.abs().clamp(min=self.cfg.turn_min)


@dataclass
class PupperVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_turn_envs: float = 0.0
  """Fraction of environments that receive a pure-yaw (turn-in-place) command."""
  turn_min: float = 0.5
  """Minimum absolute yaw rate assigned to turn-only envs [rad/s]."""

  def build(self, env: ManagerBasedRlEnv) -> PupperVelocityCommand:
    return PupperVelocityCommand(self, env)


class PupperMaxSpeedCommand(PupperVelocityCommand):
  """Flat-out forward/backward commands only: ``vx = +/-max_speed``, no y, no yaw.

  For studying what gait emerges when the only objective is speed. The magnitude
  is fixed rather than sampled, so every env is always asking for maximum effort;
  the sign is drawn 50/50 so both directions are learned.

  ``max_speed`` is deliberately set beyond what the robot can reach. It never caps
  anything -- the paired reward is unbounded in achieved speed, not a tracking
  error -- so its only role is to tell the policy, through the command in its
  observation, that the ask is "as fast as you can" rather than a specific target.
  """

  cfg: "PupperMaxSpeedCommandCfg"  # pyright: ignore[reportIncompatibleVariableOverride]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    forward = torch.rand(len(env_ids), device=self.device) < 0.5
    self.vel_command_b[env_ids, 0] = torch.where(
      forward, self.cfg.max_speed, -self.cfg.max_speed
    )
    self.vel_command_b[env_ids, 1] = 0.0
    self.vel_command_b[env_ids, 2] = 0.0


@dataclass(kw_only=True)
class PupperMaxSpeedCommandCfg(PupperVelocityCommandCfg):
  """Config for :class:`PupperMaxSpeedCommand`."""

  max_speed: float = 1.5
  """Commanded speed magnitude [m/s]. Beyond the robot's reach on purpose."""

  def build(self, env: ManagerBasedRlEnv) -> PupperMaxSpeedCommand:
    return PupperMaxSpeedCommand(self, env)


def sample_orbit_commands(
  n: int,
  radius: tuple[float, float],
  speed: tuple[float, float],
  radial: float,
  yaw_noise: float,
  device: str | torch.device = "cpu",
) -> torch.Tensor:
  """(n, 3) commands [vx, vy, wz] that circle a point ahead while facing it.

  A robot strafing sideways at ``vy`` around a point ``r`` metres straight ahead
  keeps facing it by turning at ``wz = -vy / r`` (strafing left, it turns right).
  ``speed`` is the range of |vy| (sign drawn 50/50), ``radius`` the range of r,
  ``radial`` a small fore/aft correction (|vx| <= radial, the robot holding its
  distance), and ``yaw_noise`` the std of noise added to wz (the steering
  correction that keeps the target centred).
  """
  u = torch.rand(n, 4, device=device)
  r = radius[0] + (radius[1] - radius[0]) * u[:, 0]
  vy = (speed[0] + (speed[1] - speed[0]) * u[:, 1]) * torch.where(
    u[:, 2] < 0.5, 1.0, -1.0
  )
  vx = radial * (2.0 * u[:, 3] - 1.0)
  wz = -vy / r + yaw_noise * torch.randn(n, device=device)
  return torch.stack([vx, vy, wz], dim=-1)


class OrbitVelocityCommand(PupperVelocityCommand):
  """Pupper velocity commands, a fraction of them orbits (see sample_orbit_commands)."""

  cfg: "OrbitVelocityCommandCfg"  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: "OrbitVelocityCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.is_orbit_env = torch.zeros_like(self.is_standing_env)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    r = torch.empty(len(env_ids), device=self.device)
    self.is_orbit_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_orbit_envs
    ids = env_ids[self.is_orbit_env[env_ids]]
    if len(ids) == 0:
      return
    cmd = sample_orbit_commands(
      len(ids),
      self.cfg.orbit_radius,
      self.cfg.orbit_speed,
      self.cfg.orbit_radial,
      self.cfg.orbit_yaw_noise,
      self.device,
    )
    ranges = self.cfg.ranges
    cmd[:, 0].clamp_(*ranges.lin_vel_x)
    cmd[:, 1].clamp_(*ranges.lin_vel_y)
    cmd[:, 2].clamp_(*ranges.ang_vel_z)
    self.vel_command_b[ids] = cmd
    # An orbit is a body-frame command that moves: not standing, turning in place,
    # walking straight, or a world-frame command.
    self.is_standing_env[ids] = False
    self.is_turn_env[ids] = False
    self.is_forward_env[ids] = False
    self.is_world_env[ids] = False
    self.vel_command_w[ids] = cmd


@dataclass
class OrbitVelocityCommandCfg(PupperVelocityCommandCfg):
  rel_orbit_envs: float = 0.0
  """Fraction of environments that receive an orbit command."""
  orbit_radius: tuple[float, float] = (0.6, 1.4)
  """Range of the orbit radius [m] (the game circles boxes at about 1 m)."""
  orbit_speed: tuple[float, float] = (0.1, 0.3)
  """Range of the sideways speed |vy| while orbiting [m/s]."""
  orbit_radial: float = 0.1
  """Largest fore/aft correction while orbiting [m/s]."""
  orbit_yaw_noise: float = 0.05
  """Std of the steering correction added to wz [rad/s]."""

  def build(self, env: ManagerBasedRlEnv) -> OrbitVelocityCommand:
    return OrbitVelocityCommand(self, env)
