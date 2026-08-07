"""Base class for simulated wheeled chassis robots.

Mirrors ``hex_driver_robot.robot_chassis.HexRobotChassisCallback``, but for
simulation only: the same public API (``set_chs_mit_cmd`` / ``set_chs_vel_cmd``
/ ``get_chassis_state``) is backed by the lower-level ``SimInterface`` instead
of a hardware ``Chassis`` device.

Decoupling (mirrors the arm layer)
----------------------------------
- ``HexRobotSimBase`` (in ``arm/base.py``) provides the lifecycle only.
- ``HexRobotSimChassis`` below holds the **common** chassis plumbing: step /
  work loop / command message building / state publication / odometry.
- Each chassis subclass declares its model-specific **canonical joint order**
  via ``JOINT_STATE_NAME`` and its USD config via ``_get_articulation_cfg()``.
  The wrapper resolves the USD's joints **by name** at spawn time, so it never
  assumes the USD articulation order — the command / state arrays are always
  in the robot's canonical order.
"""

import math
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from hex_util_msg.dataclass import (
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
from hex_util_runtime import HexRate, deque_helper

from ..arm.base import HexRobotSimBase, HexRobotSimParams
from ..sim.interface import ActuatorCmd
from ..utils import build_header, build_hex_jnt, build_twist, build_vector3


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass
class HexRobotSimChassisParams(HexRobotSimParams):
    """Shared parameters for simulated chassis robots (reserved, currently empty)."""


# ---------------------------------------------------------------------------
# Module-level helper — canonical-order joint resolution (by name)
# ---------------------------------------------------------------------------

def _resolve_actuator_canonical_indices(articulation, joint_state_name: list[str]):
    """Map each actuator's joints to their indices in the canonical order.

    Returns ``{actuator_name: np.ndarray[int]}`` — for every actuator that owns
    at least one canonical joint, the canonical indices of those joints, in the
    actuator's **own** joint order (which matches the ``get_joint_*`` readback
    and ``push_command`` write layout).

    Matching is purely **by joint name**, so the wrapper does not depend on the
    USD's articulation order — this is what makes the canonical order
    (e.g. the mujoco-aligned Maver X4 order) authoritative.

    Raises ``RuntimeError`` if:
      - a canonical joint is missing from the articulation,
      - a canonical joint is claimed by two actuators,
      - an actuator that owns canonical joints also owns a non-canonical joint
        (the command arrays would then be sized wrong for that actuator).
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
    """Shared sim chassis — MIT command dispatch + odometry state publication.

    Subclasses implement ``_get_articulation_cfg()`` (lazy USD config import),
    set ``CHASSIS_NAME``, and declare ``JOINT_STATE_NAME`` — the authoritative
    joint order of this robot (used for both command dispatch and state
    publication, and resolved against the USD **by joint name**).

    Usage::

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
    """

    #: Scene entity name used for spawn / articulation lookup (subclass sets).
    CHASSIS_NAME: str = ""

    #: Canonical joint order — authoritative for command & state arrays.
    #: Subclass MUST declare it (e.g. Maver X4's mujoco-aligned order).
    JOINT_STATE_NAME: list[str] = []

    def __init__(self, params: HexRobotSimChassisParams, name: str) -> None:
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
        self._deque_dict = {
            "chs_cmd": deque(maxlen=self._params.state_buffer_size),
        }
        self._cur_cmd = {"chs_cmd": None}

    # ------------------------------------------------------------------
    # init_robot — spawn articulation + resolve canonical joint layout
    # ------------------------------------------------------------------

    def init_robot(self) -> None:
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
    # work_loop — heartbeat only (sim stepping stays on the main thread)
    # ------------------------------------------------------------------

    def work_loop(self) -> None:
        """Background heartbeat.

        Sim stepping is done synchronously via :meth:`step` on the main thread
        to keep ``SimulationContext.step()`` on the event-loop thread.
        """
        rate = HexRate(self._params.ctrl_rate)
        while self.is_working():
            rate.sleep()

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
    # Command setters — same signatures as hex_driver_robot
    # ------------------------------------------------------------------

    def set_chs_mit_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Queue a direct-impedance (MIT) command.

        Args:
            cmd_dict: keys — ``jnt_pos``, ``jnt_vel``, ``mit_tau``, ``mit_kp``,
                ``mit_kd``; each an array of shape (dof,) in **canonical joint
                order** (this robot's ``JOINT_STATE_NAME``). Omitted fields
                keep the config's default PD (see ``build_hex_jnt``
                empty-array semantics below).
        """
        sim_time = self.get_sim_time()
        ts_ns = int(sim_time * 1e9) if sim_time is not None else None
        cmd = HexDcRoboChsCtrlStamped(
            header=build_header(ts_ns),
            chs_ctrl=HexDcRoboChsCtrl(
                ctrl_mode=HexDcRoboChsCtrlMode.MIT,
                jnt=build_hex_jnt(
                    pos=cmd_dict.get("jnt_pos"),
                    vel=cmd_dict.get("jnt_vel"),
                    eff=cmd_dict.get("mit_tau"),
                    kp=cmd_dict.get("mit_kp"),
                    kd=cmd_dict.get("mit_kd"),
                    dof=self._dof,
                ),
            ),
        )
        self._deque_dict["chs_cmd"].append(cmd)

    def set_chs_vel_cmd(self, cmd_dict: dict[str, Any]) -> None:
        # TODO: kinematic control (vx, vy, omega) → per-joint MIT is not
        # designed yet. User decision: raise immediately rather than no-op.
        raise NotImplementedError(
            "set_chs_vel_cmd (kinematic VEL) is not implemented; use set_chs_mit_cmd. "
            "VEL→MIT conversion is a TODO.")

    # ------------------------------------------------------------------
    # State getter — same pattern as HexRobotSimArcherY6.get_arm_state
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
        """Convert latest chassis command to per-actuator ActuatorCmds."""
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
        # NONE → no-op; VEL never arrives (setter raises NotImplementedError),
        # kept defensively.

    def _apply_chs_mit(self, jnt_info) -> None:
        """Slice a full MIT command into per-actuator ActuatorCmds.

        ``jnt_info`` is a ``HexDcBaseJntFull`` whose arrays are either size
        ``self._dof`` (in **canonical joint order**) or empty (field omitted —
        ``build_hex_jnt`` semantics). Each actuator receives only its own joint
        slice, reordered from canonical order to the actuator's local joint
        order via ``_actuator_canonical_idxs``.
        """
        def _extract_field(field_name: str) -> Optional[np.ndarray]:
            """Return the command's full-array field if present (size == dof),
            else None (field omitted → keep config default)."""
            arr = getattr(jnt_info, field_name)
            if arr is None or arr.size != self._dof:
                return None
            return np.asarray(arr, dtype=np.float32)

        command_fields = {
            "position":  _extract_field("pos"),
            "velocity":  _extract_field("vel"),
            "effort":    _extract_field("eff"),
            "stiffness": _extract_field("kp"),
            "damping":   _extract_field("kd"),
        }
        if not any(v is not None for v in command_fields.values()):
            return  # nothing to command

        for actuator_name in self._actuator_names:
            canonical_idxs = self._actuator_canonical_idxs[actuator_name]
            cmd = ActuatorCmd()
            for field, arr in command_fields.items():
                if arr is not None:
                    setattr(cmd, field, arr[canonical_idxs])
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

        # Planar odom: body-frame twist from world velocity + yaw
        # (quat is wxyz; yaw from qz,qw. World→body rotation is exact when
        #  roll/pitch≈0, which is the chassis normal operating condition.)
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
    # Subclass hook
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def _get_articulation_cfg(cls):
        """Return the USD ArticulationCfg for this chassis.

        Import happens **inside** this method (AppLauncher is already up), so
        ``hex_isaac_usd.configs`` eager-importing all configs is safe here.
        """
        raise NotImplementedError
