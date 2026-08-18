"""HexRobotSimTriggerA3 — simulated Trigger A3 chassis (3 MIT omni-wheels).

Canonical joint order (`JOINT_STATE_NAME`) is the three drive wheels only:

    [joint_1, joint_2, joint_3]

The A3 USD carries 48 extra passive ball-caster joints (no actuator). They are
not part of the canonical set and are ignored by the wrapper.

## Hardware mapping

The simulated Trigger A3 uses 3 MIT motors with direct-impedance control. 
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

#: Canonical joint order — the three drive wheels only.
JOINT_STATE_NAME = ["joint_1", "joint_2", "joint_3"]

# ---------------------------------------------------------------------------
# Velocity-level omni geometry (see `_apply_chs_vel` for the β-order detail).
# ---------------------------------------------------------------------------

#: Wheel contact radius [m].
_CHS_WHEEL_RADIUS = 0.102
#: Distance from the base centre to each wheel centre [m].
_CHS_WHEEL_DISTANCE = 0.2895
#: Velocity-tracking gain, `tau = kd·(motor_vel − qd)` (kp not used) [N·s/rad].
_CHS_KD_DEFAULT = 10.0


@dataclass
class HexRobotSimTriggerA3Params(HexRobotSimChassisParams):
    """Parameters for the simulated Trigger A3 chassis."""


class HexRobotSimTriggerA3(HexRobotSimChassis):
    """Simulated Trigger A3 (3 omni-wheels, single drive actuator, MIT)."""

    CHASSIS_NAME = "trigger_a3"
    JOINT_STATE_NAME = JOINT_STATE_NAME

    def __init__(
        self,
        params: HexRobotSimTriggerA3Params = HexRobotSimTriggerA3Params(),
    ) -> None:
        """Create the Trigger A3 sim chassis with default params."""
        super().__init__(params=params, name="Trigger_a3")

    # ------------------------------------------------------------------
    # Model-specific command setter — direct-impedance (MIT)
    # ------------------------------------------------------------------

    def set_chs_mit_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Queue a direct-impedance (MIT) command for the 3 chassis motors.

        Args:
            cmd_dict: keys — `jnt_pos`, `jnt_vel`, `mit_tau`, `mit_kp`,
                `mit_kd`; each an array of shape (3,) in **canonical joint
                order** (`JOINT_STATE_NAME`). Omitted fields keep the config's
                default PD (see `build_hex_jnt` empty-array semantics).

        Raises:
            ValueError: If any provided array has shape != (3,).
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
    # Kinematic VEL → MIT solver (velocity-level omni)
    # ------------------------------------------------------------------

    def _apply_chs_vel(self, vel: HexDcBaseTwist) -> None:
        """Velocity-level 3-omni solver: (vx, vy, omega) → per-joint MIT targets.

        A fixed 3×3 inverse Jacobian built from the wheel azimuths
        β = [π/3, π, −π/3], row-aligned to the canonical joint order — the
        middle wheel (β = π) sits on joint_2. The leading −1 reflects the
        wheel-axis sign convention.

        All three wheels are commanded as joint-velocity targets with
        `kp=0` + `kd=_CHS_KD_DEFAULT`, so each ImplicitActuator produces the
        reference velocity tracker `tau = kd·(ω_tgt − ω)` (no position term).

        Args:
            vel: Velocity twist command (linear x/y, angular z).
        """
        vx, vy, omega = float(vel.linear.x), float(vel.linear.y), float(vel.angular.z)
        cmd = np.array([vx, vy, omega])

        # Azimuths β=[π/3, π, −π/3]; see the docstring for the row-order detail.
        beta = np.array([math.pi / 3, math.pi, -math.pi / 3])
        s, c = np.sin(beta), np.cos(beta)
        jac_inv = -1.0 / _CHS_WHEEL_RADIUS * np.array([
            [-s[0], c[0], _CHS_WHEEL_DISTANCE],
            [-s[1], c[1], _CHS_WHEEL_DISTANCE],
            [-s[2], c[2], _CHS_WHEEL_DISTANCE],
        ])
        motor_vel = jac_inv @ cmd  # canonical order [joint_1, joint_2, joint_3]

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
        """Return the Trigger A3 USD `ArticulationCfg`."""
        from hex_isaac_usd.configs import HEX_ISAAC_USD_TRIGGER_A3_CFG
        return HEX_ISAAC_USD_TRIGGER_A3_CFG
