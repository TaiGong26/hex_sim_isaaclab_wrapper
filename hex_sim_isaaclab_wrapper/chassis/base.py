"""Shared base for simulated wheeled chassis robots.

The base holds the **common** plumbing — lifecycle, per-name
joint resolution, state publication / odometry — and dispatches the
queued chassis command by `ctrl_mode` (MIT direct-impedance / VEL
kinematic twist). Model-specific command setters (e.g. the direct-
impedance `set_chs_mit_cmd`) and the kinematic VEL→joint solver live in
each robot subclass.

## Decoupling

- `HexRobotSimBase` (in `arm/base.py`) provides the lifecycle only.
- `HexRobotSimChassis` below is a **thin** base: spawn, per-name joint
  resolution, step / work loop, command dispatch (by `ctrl_mode`),
  state reading + odometry, shared getters.
- Each chassis subclass declares `JOINT_STATE_NAME` (the canonical joint
  order), owns its `set_*` command setters, and implements the abstract
  `_apply_chs_vel` kinematic solver (may also override the shared
  `_apply_chs_mit` scatter if ever needed).
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from hex_util_msg.dataclass import (
    HexDcBaseJntFull,
    HexDcBaseJntState,
    HexDcBaseOdometry,
    HexDcBasePose,
    HexDcBaseQuaternion,
    HexDcBaseTwist,
    HexDcRoboChsCtrl,
    HexDcRoboChsCtrlMode,
    HexDcRoboChsCtrlStamped,
    HexDcRoboChsState,
    HexDcRoboChsStateStamped,
)
from hex_util_runtime import deque_helper

from ..arm.base import HexRobotSimBase, HexRobotSimParams
from ..sim.interface import ActuatorCmd
from ..utils import build_header, build_hex_jnt, build_twist, build_vector3

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, ArticulationCfg


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass
class HexRobotSimChassisParams(HexRobotSimParams):
    """Shared parameters for simulated chassis robots (reserved, currently empty)."""


# ---------------------------------------------------------------------------
# Module-level helper — canonical-order joint resolution (by name)
# ---------------------------------------------------------------------------

def _resolve_actuator_canonical_indices(
    articulation: Articulation, joint_state_name: list[str]
) -> dict[str, np.ndarray]:
    """Map each actuator's joints to their indices in the canonical order.

    Returns `{actuator_name: np.ndarray[int]}` — for every actuator that owns
    at least one canonical joint, the canonical indices of those joints, in
    the actuator's **own** joint order (which matches the `get_joint_*`
    readback and `push_command` write layout).

    Matching is purely **by joint name**, so the wrapper does not depend on
    the USD's articulation order.

    Args:
        articulation:     Spawned articulation whose joints are matched.
        joint_state_name: Canonical joint names of this robot.

    Returns:
        Actuator name → canonical indices of its joints, in the actuator's
        own joint order (matches the `get_joint_*` readback layout).

    Raises:
        RuntimeError: If a canonical joint is missing from the articulation,
            claimed by two actuators, or an actuator that owns canonical
            joints also owns a non-canonical joint (the command arrays would
            then be sized wrong for that actuator).
    """
    articulation_joint_names = list(articulation.joint_names)
    missing = [n for n in joint_state_name if n not in articulation_joint_names]
    if missing:
        raise RuntimeError(
            f"Articulation is missing canonical joints {missing}; "
            f"articulation joints: {articulation_joint_names}")

    name_to_canonical = {name: i for i, name in enumerate(joint_state_name)}

    actuator_canonical: dict[str, np.ndarray] = {}
    for actuator_name, actuator in articulation.actuators.items():
        local_names = list(actuator.joint_names)
        local_canonical = [name_to_canonical[n] for n in local_names
                           if n in name_to_canonical]
        if not local_canonical:
            continue  # actuator owns only non-canonical joints — ignore it
        extra = [n for n in local_names if n not in name_to_canonical]
        if extra:
            raise RuntimeError(
                f"Actuator '{actuator_name}' owns non-canonical joints {extra}; "
                f"canonical joint set: {joint_state_name}")
        actuator_canonical[actuator_name] = np.asarray(
            local_canonical, dtype=np.int64)

    covered = np.concatenate(list(actuator_canonical.values())) \
        if actuator_canonical else np.array([], dtype=np.int64)
    if covered.size != len(joint_state_name) \
            or np.unique(covered).size != len(joint_state_name):
        raise RuntimeError(
            f"Canonical joints are not uniquely covered by the actuators: "
            f"expected {len(joint_state_name)} canonical indices, "
            f"resolved {covered.tolist()}")

    return actuator_canonical


# ---------------------------------------------------------------------------
# Chassis base
# ---------------------------------------------------------------------------

class HexRobotSimChassis(HexRobotSimBase):
    """Thin shared sim chassis — lifecycle + state publication + command dispatch.

    Subclasses implement `_get_articulation_cfg()` (lazy USD config import),
    set `CHASSIS_NAME`, declare `JOINT_STATE_NAME` (the authoritative joint
    order), own their `set_*` command setters, and implement the abstract
    `_apply_chs_vel` kinematic solver.

    Example:

    ```python
    params = HexRobotSimMaverX4Params(...)
    robot = HexRobotSimMaverX4(params)
    robot.start()
    while robot.is_working():
        robot.set_chs_mit_cmd({
            "jnt_pos": np.zeros(8), "jnt_vel": np.zeros(8),
            "mit_tau": np.zeros(8), "mit_kp": np.full(8, 400.0),
            "mit_kd": np.full(8, 20.0),
        })
        robot.step()
        st = robot.get_chassis_state()
        time.sleep(1.0 / params.ctrl_rate)
    robot.stop()
    ```
    """

    #: Scene entity name used for spawn / articulation lookup (subclass sets).
    CHASSIS_NAME: str = ""

    #: Canonical joint order — authoritative for command & state arrays.
    #: Subclass MUST declare it (e.g. Maver X4's wheel-first order).
    JOINT_STATE_NAME: list[str] = []

    def __init__(self, params: HexRobotSimChassisParams, name: str) -> None:
        """Set up user-facing state deques/callbacks and chassis buffers."""
        super().__init__(params=params, name=name)

        # User-facing state deques / callbacks
        self._deque_user: dict[str, deque] = {
            "chs_state": deque(maxlen=params.state_buffer_size),
        }
        self._callbacks = {
            "chs_state": self._deque_user["chs_state"].append,
        }

        # Resolved in init_robot()
        self._dof: int = 0
        # Actuator names, ordered by first canonical index (iteration order for
        # scatter/gather). The publish order is canonical, independent of this.
        self._actuator_names: list[str] = []
        # Actuator name → canonical indices of its joints (in actuator-local
        # joint order), resolved by name at spawn time.
        self._actuator_canonical_idxs: dict[str, np.ndarray] = {}
        self._deque_dict: dict[str, Optional[deque]] = {}
        self._cur_cmd: dict[str, Optional[Any]] = {}
        self._cur_state: dict[str, Optional[dict]] = {}

    # ------------------------------------------------------------------
    # init_vars — non-DOF-dependent init (called by start() before init_robot)
    # ------------------------------------------------------------------

    def init_vars(self) -> None:
        """Initialize the command deque and current-command buffer."""
        self._deque_dict = {
            "chs_cmd": deque(maxlen=self._params.state_buffer_size),
        }
        self._cur_cmd = {"chs_cmd": None}

    # ------------------------------------------------------------------
    # init_robot — spawn articulation + resolve canonical joint layout
    # ------------------------------------------------------------------

    def init_robot(self) -> None:
        """Spawn the articulation and resolve the canonical joint layout."""
        # 1. Isaac Lab chassis interface — first, to satisfy AppLauncher
        from ..sim.isaaclab_chassis_interface import IsaacLabChassisInterface

        cli_args = ["--headless"] if bool(self._params.isaac_headless) else []
        sim = IsaacLabChassisInterface()
        sim.initialize(cli_args=cli_args, device=self._params.torch_device,
                       num_envs=self._params.sim_num_envs, dt=self._params.ctrl_rate,
                       render_rate=self._params.render_rate,
                       sim_env=self._params.sim_env)
        self._sim_interface = sim

        # 2. ArticulationCfg — safe only after AppLauncher is up
        articulation_cfg = self._get_articulation_cfg()

        # 3. Spawn + one step to populate data buffers
        sim.spawn_robot(self.CHASSIS_NAME, articulation_cfg)
        sim.step()

        # 4. Resolve this robot's canonical joint order (JOINT_STATE_NAME)
        #    against the spawned articulation, **by joint name**. The USD
        #    articulation order never matters — only that every canonical joint
        #    exists and is uniquely owned by one actuator.
        articulation = sim._scene[self.CHASSIS_NAME]
        if not self.JOINT_STATE_NAME:
            raise AssertionError(
                f"{type(self).__name__} must declare JOINT_STATE_NAME "
                "(the canonical joint order)")
        self._actuator_canonical_idxs = _resolve_actuator_canonical_indices(
            articulation, self.JOINT_STATE_NAME)
        self._actuator_names = sorted(
            self._actuator_canonical_idxs.keys(),
            key=lambda name: int(self._actuator_canonical_idxs[name][0]))

        # 5. DOF = number of joints in the canonical order (= the actuated
        #    joints). Unactuated USD joints (e.g. the A3's 48 passive
        #    ball-casters) are outside the canonical set and stay untouched.
        self._dof = len(self.JOINT_STATE_NAME)

        # 6. State buffers
        self._cur_state = {
            "chs": {
                "jnt_pos": np.zeros(self._dof),
                "jnt_vel": np.zeros(self._dof),
                "jnt_eff": np.zeros(self._dof),
            },
        }
        self.logi(
            f"Chassis '{self.CHASSIS_NAME}' dof={self._dof} "
            f"joint_state_name={self.JOINT_STATE_NAME} "
            f"actuator_canonical_idxs={self._actuator_canonical_idxs}")

    # ------------------------------------------------------------------
    # step — process → sim step → publish (call from main thread)
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Process commands → step simulation → publish state."""
        self._process_chs_cmd()
        self._sim_interface.step()
        try:
            self._update_chs_state()
        except Exception:
            self.loge("State callback failure", exc_info=True)

    # ------------------------------------------------------------------
    # Command setter — kinematic VEL (per-robot solver)
    # ------------------------------------------------------------------

    def set_chs_vel_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Queue a kinematic velocity command (body-frame twist).

        The per-joint conversion is owned by the robot's `_apply_chs_vel`;
        this setter only packages the twist into a `VEL`-mode chassis command.

        Args:
            cmd_dict: dict with optional keys:
                ts_ns : int, timestamp (default: simulation time)
                vx    : float, linear velocity x [m/s]
                vy    : float, linear velocity y [m/s]
                omega : float, angular velocity z [rad/s]
        """
        sim_time = self.get_sim_time()
        ts_ns = cmd_dict.get("ts_ns")
        if ts_ns is None:
            ts_ns = int(sim_time * 1e9) if sim_time is not None else None
        cmd = HexDcRoboChsCtrlStamped(
            header=build_header(ts_ns),
            chs_ctrl=HexDcRoboChsCtrl(
                ctrl_mode=HexDcRoboChsCtrlMode.VEL,
                jnt=build_hex_jnt(dof=0),
                vel=build_twist(
                    linear=(cmd_dict.get("vx", 0.0), cmd_dict.get("vy", 0.0), 0.0),
                    angular=(0.0, 0.0, cmd_dict.get("omega", 0.0)),
                ),
            ),
        )
        self._deque_dict["chs_cmd"].append(cmd)

    # ------------------------------------------------------------------
    # State getter
    # ------------------------------------------------------------------

    def get_chassis_state(self, latest: bool = True) -> Optional[HexDcRoboChsStateStamped]:
        """Return latest chassis state (jnt + odom), or None."""
        return deque_helper(self._deque_user["chs_state"], latest=latest)

    def get_sim_time(self) -> Optional[float]:
        """Return sim time in seconds (Isaac Lab sim time, not wall-clock)."""
        if self._sim_interface is None:
            return None
        return self._sim_interface.get_sim_time()

    # ------------------------------------------------------------------
    # Internal — command processing (main-thread context)
    # ------------------------------------------------------------------

    def _process_chs_cmd(self) -> None:
        """Dispatch the latest chassis command by `ctrl_mode`.

        - `MIT`  → `_apply_chs_mit` (direct-impedance scatter, shared)
        - `VEL`  → `_apply_chs_vel` (per-robot kinematic solver)
        - other  → no-op

        Subclasses may override `_apply_chs_*` to extend handling; the
        mode switch itself stays here.
        """
        temp = deque_helper(self._deque_dict["chs_cmd"], latest=True)
        if temp is not None:
            self._cur_cmd["chs_cmd"] = temp

        cmd_stamped = self._cur_cmd["chs_cmd"]
        if cmd_stamped is None:
            return

        ctrl = cmd_stamped.chs_ctrl
        mode = ctrl.ctrl_mode
        if mode == HexDcRoboChsCtrlMode.MIT:
            self._apply_chs_mit(ctrl.jnt)
        elif mode == HexDcRoboChsCtrlMode.VEL:
            self._apply_chs_vel(ctrl.vel)

    def _apply_chs_mit(self, jnt_info: HexDcBaseJntFull) -> None:
        """Slice a full MIT command into per-actuator ActuatorCmds.

        Shared scatter primitive — robot-agnostic. `jnt_info` is a
        `HexDcBaseJntFull` whose arrays are either size `self._dof` (in
        **canonical joint order**) or empty (field omitted — `build_hex_jnt`
        semantics). Each actuator receives only its own joint slice, reordered
        from canonical order to the actuator's local joint order via
        `_actuator_canonical_idxs`.

        Args:
            jnt_info: Full canonical-order MIT command fields.
        """
        dof = self._dof
        pos = np.asarray(jnt_info.pos, dtype=np.float32) \
            if jnt_info.pos is not None and jnt_info.pos.size == dof else np.zeros(dof)
        vel = np.asarray(jnt_info.vel, dtype=np.float32) \
            if jnt_info.vel is not None and jnt_info.vel.size == dof else np.zeros(dof)
        eff = np.asarray(jnt_info.eff, dtype=np.float32) \
            if jnt_info.eff is not None and jnt_info.eff.size == dof else np.zeros(dof)
        kp = np.asarray(jnt_info.kp, dtype=np.float32) \
            if jnt_info.kp is not None and jnt_info.kp.size == dof else np.zeros(dof)
        kd = np.asarray(jnt_info.kd, dtype=np.float32) \
            if jnt_info.kd is not None and jnt_info.kd.size == dof else np.zeros(dof)

        for actuator_name in self._actuator_names:
            canonical_idxs = self._actuator_canonical_idxs[actuator_name]
            cmd = ActuatorCmd(
                position=pos[canonical_idxs],
                velocity=vel[canonical_idxs],
                effort=eff[canonical_idxs],
                stiffness=kp[canonical_idxs],
                damping=kd[canonical_idxs],
            )
            self._sim_interface.push_command(actuator=actuator_name, cmd=cmd)

    # ------------------------------------------------------------------
    # Internal — state update and publish (main-thread context)
    # ------------------------------------------------------------------

    def _read_chs_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read per-actuator joint state, assembled into canonical order."""
        sim = self._sim_interface
        pos = np.zeros(self._dof)
        vel = np.zeros(self._dof)
        eff = np.zeros(self._dof)
        for actuator_name in self._actuator_names:
            canonical_idxs = self._actuator_canonical_idxs[actuator_name]
            pos[canonical_idxs] = sim.get_joint_positions(actuator_name)
            vel[canonical_idxs] = sim.get_joint_velocities(actuator_name)
            eff[canonical_idxs] = sim.get_joint_efforts(actuator_name)
        return pos, vel, eff

    def _update_chs_state(self) -> None:
        """Read joint state + odom from sim → build msg → push to callback deque."""
        sim = self._sim_interface

        # Joint state in canonical order (this robot's JOINT_STATE_NAME)
        pos, vel, eff = self._read_chs_joint_state()
        self._cur_state["chs"]["jnt_pos"][:] = pos
        self._cur_state["chs"]["jnt_vel"][:] = vel
        self._cur_state["chs"]["jnt_eff"][:] = eff

        root_pos, root_quat = sim.get_root_pose()           # (3,), (4,) wxyz
        lin_vel_w, ang_vel_w = sim.get_root_velocity()      # world frame
        yaw = 2.0 * math.atan2(root_quat[3], root_quat[0])  # z, w
        vx_b = lin_vel_w[0] * math.cos(yaw) + lin_vel_w[1] * math.sin(yaw)
        vy_b = -lin_vel_w[0] * math.sin(yaw) + lin_vel_w[1] * math.cos(yaw)
        omega_b = ang_vel_w[2]

        odom_pose = HexDcBasePose(
            position=build_vector3(x=float(root_pos[0]), y=float(root_pos[1]), z=0.0),
            orientation=HexDcBaseQuaternion(
                x=0.0, y=0.0,
                z=np.sin(yaw * 0.5), w=np.cos(yaw * 0.5)),
        )
        odom_twist = build_twist(
            linear=(vx_b, vy_b, 0.0), angular=(0.0, 0.0, omega_b))

        sim_time = self.get_sim_time()
        ts_ns = int(sim_time * 1e9) if sim_time is not None else None
        state_msg = HexDcRoboChsStateStamped(
            header=build_header(ts_ns),
            chs_state=HexDcRoboChsState(
                jnt=HexDcBaseJntState(position=pos, velocity=vel, effort=eff),
                odom=HexDcBaseOdometry(pose=odom_pose, twist=odom_twist),
            ),
        )
        self._callbacks["chs_state"](state_msg)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def _get_articulation_cfg(cls) -> ArticulationCfg:
        """Return the USD `ArticulationCfg` for this chassis.

        Import happens **inside** this method (AppLauncher is already up), so
        eager-importing all configs here is safe.
        """
        raise NotImplementedError

    @abstractmethod
    def _apply_chs_vel(self, vel: HexDcBaseTwist) -> None:
        """Kinematic VEL → per-joint command solver (per-robot).

        Converts a velocity twist `vel` (vx / vy / omega) into this robot's
        per-joint MIT targets (e.g. the Maver swerve inverse Jacobian or the
        Trigger omni Jacobian).

        Args:
            vel: Velocity twist command (linear x/y, angular z).
        """
        raise NotImplementedError
