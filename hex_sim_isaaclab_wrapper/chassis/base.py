"""Base classes for simulated wheeled chassis robots.

Mirrors ``hex_driver_robot.robot_chassis.HexRobotChassisCallback``, but for
simulation only: the same public API (``set_chs_mit_cmd`` / ``set_chs_vel_cmd``
/ ``get_chassis_state``) is backed by the lower-level ``SimInterface`` instead
of a hardware ``Chassis`` device.
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
from ..utils import build_header, build_hex_jnt, build_twist, build_vector3, torch_to_numpy


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass
class HexRobotSimChassisParams(HexRobotSimParams):
    """Shared parameters for simulated chassis robots (reserved, currently empty)."""


# ---------------------------------------------------------------------------
# Chassis base
# ---------------------------------------------------------------------------

class HexRobotSimChassis(HexRobotSimBase):
    """Shared sim chassis — MIT command dispatch + odometry state publication.

    Subclasses implement ``_get_articulation_cfg()`` (lazy USD config import)
    and override ``CHASSIS_NAME``.

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
        # Actuator names, ordered by first global joint index (articulation
        # joint order); used as the iteration order for gather/scatter.
        self._actuator_names: list[str] = []
        # Actuator name → the global indices of its joints (int64 array).
        self._actuator_joint_indices: dict[str, np.ndarray] = {}
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
    # init_robot — spawn articulation + resolve DOF/actuator layout
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

        # 4. Map each actuator to its global joint indices; order the actuator
        #    names by first joint index ascending (articulation joint order),
        #    never by dict insertion order.
        articulation = sim._scene[self.CHASSIS_NAME]
        actuator_names = list(articulation.actuators.keys())
        self._actuator_joint_indices = {
            actuator_name: torch_to_numpy(
                articulation.actuators[actuator_name].joint_indices).astype(np.int64)
            for actuator_name in actuator_names
        }
        self._actuator_names = sorted(
            actuator_names,
            key=lambda name: int(self._actuator_joint_indices[name][0]))

        # 5. DOF = number of *actuated* joints (sum across actuator groups).
        #    The articulation may contain extra unactuated joints (e.g. the A3
        #    USD has 27 joints, only 3 actuated by ``drive_wheels``). This
        #    matches the driver's ``motor_count`` (A3=3, X4=8).
        self._dof = int(np.sum(
            [self._actuator_joint_indices[name].size for name in self._actuator_names]))

        # 6. Consistency — actuator joint indices must be non-overlapping and
        #    lie within the articulation's joint array.
        all_joint_indices = np.concatenate(
            [self._actuator_joint_indices[name] for name in self._actuator_names])
        if np.unique(all_joint_indices).size != all_joint_indices.size:
            detail = ", ".join(
                f"{name}={self._actuator_joint_indices[name].tolist()}"
                for name in self._actuator_names)
            raise AssertionError(f"Actuator joint indices overlap: {detail}")
        articulation_dof = int(articulation.data.joint_pos.shape[-1])
        if int(np.max(all_joint_indices)) >= articulation_dof:
            detail = ", ".join(
                f"{name}={self._actuator_joint_indices[name].tolist()}"
                for name in self._actuator_names)
            raise AssertionError(
                f"Actuator joint indices out of range [0, {articulation_dof}): {detail}")

        # 7. State buffers
        self._cur_state = {
            "chs": {
                "jnt_pos": np.zeros(self._dof),
                "jnt_vel": np.zeros(self._dof),
                "jnt_eff": np.zeros(self._dof),
            },
        }
        self.logi(
            f"Chassis '{self.CHASSIS_NAME}' dof={self._dof} "
            f"actuators={self._actuator_names} "
            f"joint_indices={self._actuator_joint_indices}")

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
                ``mit_kd``; each an array of shape (dof,) in **articulation
                joint order** (X4: steering 0-3, drive 4-7; A3: joint_1..3 =
                0-2).  Omitted fields keep the config's default PD (see
                ``build_hex_jnt`` empty-array semantics below).
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
        ``self._dof`` (in articulation joint order) or empty (field omitted —
        ``build_hex_jnt`` semantics).  Each actuator receives only its own
        joint slice, gathered by its global joint indices.
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
            joint_indices = self._actuator_joint_indices[actuator_name]
            cmd = ActuatorCmd()
            for field, arr in command_fields.items():
                if arr is not None:
                    setattr(cmd, field, arr[joint_indices])
            self._sim_interface.push_command(actuator=actuator_name, cmd=cmd)

    # ------------------------------------------------------------------
    # Internal — state update and publish (main-thread context)
    # ------------------------------------------------------------------

    def _update_chs_state(self) -> None:
        """Read joint state + odom from sim → build msg → push to callback deque."""
        sim = self._sim_interface

        # Joint state in articulation joint order (scatter by actuator jids)
        pos = np.zeros(self._dof)
        vel = np.zeros(self._dof)
        eff = np.zeros(self._dof)
        for actuator_name in self._actuator_names:
            joint_indices = self._actuator_joint_indices[actuator_name]
            pos[joint_indices] = sim.get_joint_positions(actuator_name)
            vel[joint_indices] = sim.get_joint_velocities(actuator_name)
            eff[joint_indices] = sim.get_joint_efforts(actuator_name)

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
