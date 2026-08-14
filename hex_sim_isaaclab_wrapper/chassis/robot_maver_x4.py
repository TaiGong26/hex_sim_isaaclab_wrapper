"""HexRobotSimMaverX4 — simulated Maver X4 chassis (8 MIT motors).

Canonical joint order (`JOINT_STATE_NAME`) — wheel-first — must not be
changed without updating all of them:

    [joint_wheel1, joint_yaw1, joint_wheel2, joint_yaw2,
     joint_wheel3, joint_yaw3, joint_wheel4, joint_yaw4]

- drive (wheels)  → canonical indices [0, 2, 4, 6]
- steering (yaw)  → canonical indices [1, 3, 5, 7]

The wrapper matches the USD's joints **by name** at spawn time, so the USD
articulation order itself never matters.

## Model-specific ownership

This class owns its direct-impedance MIT command (`set_chs_mit_cmd`) and its
dispatch (`_dispatch_chassis_command`); the shared chassis base only provides
the lifecycle / state publication / odometry and a generic dispatch skeleton.
The per-motor speed API of the Trigger A3 lr1 variant is blocked here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from hex_util_msg.dataclass import (
    HexDcBaseTwist,
    HexDcRoboChsCtrl,
    HexDcRoboChsCtrlMode,
    HexDcRoboChsCtrlStamped,
)

from .base import HexRobotSimChassis, HexRobotSimChassisParams
from ..utils import build_header, build_hex_jnt

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

#: Canonical joint order — wheel-first (each module's wheel before its yaw).
JOINT_STATE_NAME = [
    "joint_wheel1", "joint_yaw1", "joint_wheel2", "joint_yaw2",
    "joint_wheel3", "joint_yaw3", "joint_wheel4", "joint_yaw4",
]

#: Canonical indices of the four steering (yaw) joints, derived by **name**
#: from `JOINT_STATE_NAME`. The interface arrays keep the canonical index
#: order unchanged; only the internal solver lookups go by name so a future
#: `JOINT_STATE_NAME` reordering cannot silently misalign.
_YAW_IDX = np.array(
    [i for i, n in enumerate(JOINT_STATE_NAME) if n.startswith("joint_yaw")],
    dtype=np.int64)

#: Canonical indices of the four drive (wheel) joints, derived by name.
_WHEEL_IDX = np.array(
    [i for i, n in enumerate(JOINT_STATE_NAME) if n.startswith("joint_wheel")],
    dtype=np.int64)

# ---------------------------------------------------------------------------
# Velocity-level swerve geometry — wheel collision radius 0.0625, module yaw
# (±0.212, ±0.14) / wheel (±0.192/0.232, ±0.14) → track/base/bias below.
# ---------------------------------------------------------------------------

#: Distance between the left and right module yaw axes [m].
_CHS_TRACK_WIDTH = 0.28
#: Distance between the front and rear module yaw axes [m].
_CHS_WHEEL_BASE = 0.424
#: Wheel contact radius [m].
_CHS_WHEEL_RADIUS = 0.0625
#: Wheel-centre inset behind the yaw axis (along the module x) [m].
_CHS_BIAS = 0.02
#: Velocity-tracking gain, `tau = kd·(motor_vel − qd)` (kp not used) [N·s/rad].
_CHS_KD_DEFAULT = 10.0


@dataclass
class HexRobotSimMaverX4Params(HexRobotSimChassisParams):
    """Parameters for the simulated Maver X4 chassis."""


class _UnsupportedPerMotorSpeedCommand:
    """Descriptor that blocks the Trigger lr1-only API from Maver APIs."""

    def __get__(self, instance, owner):
        raise AttributeError(
            "Maver chassis does not support set_chs_per_motor_spd_cmd; "
            "use set_chs_mit_cmd instead"
        )


class HexRobotSimMaverX4(HexRobotSimChassis):
    """Simulated Maver X4 (4 steering + 4 drive MIT motors)."""

    CHASSIS_NAME = "maver_x4"
    JOINT_STATE_NAME = JOINT_STATE_NAME

    set_chs_per_motor_spd_cmd = _UnsupportedPerMotorSpeedCommand()

    def __init__(
        self,
        params: HexRobotSimMaverX4Params = HexRobotSimMaverX4Params(),
    ) -> None:
        """Create the Maver X4 sim chassis with default params."""
        super().__init__(params=params, name="Maver_x4")

    # ------------------------------------------------------------------
    # Model-specific command setter — direct-impedance (MIT)
    # ------------------------------------------------------------------

    def set_chs_mit_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Queue a direct-impedance (MIT) command for the 8 chassis motors.

        Args:
            cmd_dict: keys — `jnt_pos`, `jnt_vel`, `mit_tau`, `mit_kp`,
                `mit_kd`; each an array of shape (8,) in **canonical joint
                order** (`JOINT_STATE_NAME`). Omitted fields keep the config's
                default PD (see `build_hex_jnt` empty-array semantics).

        Raises:
            ValueError: If any provided array has shape != (8,).
        """
        dof = self._dof

        def _array(key: str) -> Optional[np.ndarray]:
            value = cmd_dict.get(key)
            if value is None:
                return None
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape != (dof,):
                raise ValueError(f"{key} must have shape ({dof},), got {arr.shape}")
            return arr

        sim_time = self.get_sim_time()
        ts_ns = int(sim_time * 1e9) if sim_time is not None else None
        cmd = HexDcRoboChsCtrlStamped(
            header=build_header(ts_ns),
            chs_ctrl=HexDcRoboChsCtrl(
                ctrl_mode=HexDcRoboChsCtrlMode.MIT,
                jnt=build_hex_jnt(
                    pos=_array("jnt_pos"),
                    vel=_array("jnt_vel"),
                    eff=_array("mit_tau"),
                    kp=_array("mit_kp"),
                    kd=_array("mit_kd"),
                    dof=dof,
                ),
            ),
        )
        self._deque_dict["chs_cmd"].append(cmd)

    # ------------------------------------------------------------------
    # Model-specific dispatch
    # ------------------------------------------------------------------

    def _dispatch_chassis_command(self, cmd: object) -> bool:
        """Dispatch the direct-impedance (MIT) command supported by Maver motors."""
        if not isinstance(cmd, HexDcRoboChsCtrlStamped):
            return False
        if cmd.chs_ctrl.ctrl_mode != HexDcRoboChsCtrlMode.MIT:
            return False
        self._apply_chs_mit(cmd.chs_ctrl.jnt)
        return True

    # ------------------------------------------------------------------
    # Kinematic VEL → MIT solver (velocity-level swerve)
    # ------------------------------------------------------------------

    def _apply_chs_vel(self, vel: HexDcBaseTwist) -> None:
        """Velocity-level 4WIS solver: (vx, vy, omega) → per-joint MIT targets.

        The 8×3 inverse Jacobian is rebuilt each step from the current steering
        angles. The yaw/wheel rows are looked up **by name** from
        `JOINT_STATE_NAME` (`_YAW_IDX` / `_WHEEL_IDX`), so the solver is immune
        to a future canonical reordering; the interface arrays keep the
        canonical index order unchanged.

        Both steering and drive are commanded as joint-velocity targets with
        `kp=0` + `kd=_CHS_KD_DEFAULT`, so each ImplicitActuator produces the
        reference velocity tracker `tau = kd·(ω_tgt − ω)` (no position term).

        Args:
            vel: Velocity twist command (linear x/y, angular z).
        """
        vx, vy, omega = float(vel.linear.x), float(vel.linear.y), float(vel.angular.z)
        cmd = np.array([vx, vy, omega])

        # Per-module geometry (all 4 modules are identical).
        temp_beta = math.atan2(_CHS_TRACK_WIDTH, _CHS_WHEEL_BASE)
        beta = np.array(
            [temp_beta, math.pi - temp_beta, temp_beta - math.pi, -temp_beta])
        wheel_distance = 0.5 * math.hypot(_CHS_TRACK_WIDTH, _CHS_WHEEL_BASE)

        # Current steering angles — looked up by name (`_YAW_IDX`), canonical
        # order [yaw1, yaw2, yaw3, yaw4].
        theta = self._cur_state["chs"]["jnt_pos"][_YAW_IDX]
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_theta_beta = np.sin(theta - beta)
        cos_theta_beta = np.cos(theta - beta)

        mat_wheel = np.column_stack(
            (cos_theta, sin_theta, wheel_distance * sin_theta_beta)
        ) / _CHS_WHEEL_RADIUS
        mat_yaw = np.column_stack(
            (-sin_theta, cos_theta,
             wheel_distance * cos_theta_beta - _CHS_BIAS)
        ) / _CHS_BIAS

        jac_inv = np.empty((8, 3))
        jac_inv[_YAW_IDX, :] = mat_yaw
        jac_inv[_WHEEL_IDX, :] = mat_wheel
        motor_vel = jac_inv @ cmd  # canonical order [wheel1, yaw1, ..., wheel4, yaw4]

        jnt_info = build_hex_jnt(
            pos=np.zeros(self._dof), vel=motor_vel, eff=np.zeros(self._dof),
            kp=np.zeros(self._dof), kd=np.full(self._dof, _CHS_KD_DEFAULT),
            dof=self._dof,
        )
        self._apply_chs_mit(jnt_info)
        

    # ------------------------------------------------------------------
    # USD config
    # ------------------------------------------------------------------

    @classmethod
    def _get_articulation_cfg(cls) -> ArticulationCfg:
        """Return the Maver X4 USD `ArticulationCfg`."""
        from hex_isaac_usd.configs import HEX_ISAAC_USD_MAVER_X4_CFG
        return HEX_ISAAC_USD_MAVER_X4_CFG
