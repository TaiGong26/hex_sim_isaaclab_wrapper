"""HexRobotSimArcherY6 — simulation-only Archer Y6 robot.

API aligned with ``hex_driver_robot.robot_archer_y6.HexRobotArcherY6``
(but only simulation-relevant methods, no hardware).

V1 scope: 6-DOF arm + optional GR100 gripper (8-DOF USD).
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from hex_util_msg.dataclass import (
    HexDcBaseJntState,
    HexDcBasePose,
    HexDcBaseQuaternion,
    HexDcBaseVector3,
    HexDcRoboArmCtrl,
    HexDcRoboArmCtrlMode,
    HexDcRoboArmCtrlStamped,
    HexDcRoboArmState,
    HexDcRoboArmStateStamped,
    HexDcRoboGripCtrl,
    HexDcRoboGripCtrlMode,
    HexDcRoboGripCtrlStamped,
    HexDcRoboGripState,
    HexDcRoboGripStateStamped,
)
from hex_util_runtime import deque_helper, ns_now

from ..utils import build_header, build_hex_jnt, torch_to_numpy
from .base import HexRobotSimBase, HexRobotSimParams


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Stiffness / damping from HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG arm actuator.
# Used for effort estimation in sim (ImplicitActuator PD law).
_ARM_STIFFNESS = np.array([400.0, 400.0, 500.0, 200.0, 100.0, 100.0])
_ARM_DAMPING = np.array([20.0, 20.0, 20.0, 20.0, 2.0, 2.0])

# Grip actuator from GR100 config
_GRIP_STIFFNESS = np.array([100.0, 100.0])
_GRIP_DAMPING = np.array([10.0, 10.0])

# USD config aliases — values match key names in hex_isaac_usd.configs
_GRIP_TO_USD: dict[str, str] = {
    "empty": "HEX_ISAAC_USD_ARCHER_Y6_CFG",
    "gp80": "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG",
    "gr100": "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG",
    "gp100": "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG",
}

# Sim USD always has 2 grip joints (J1, J2). User commands 1-DOF values;
# we replicate to both joints internally.
_SIM_GRIP_DOF = 2


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass
class HexRobotSimArcherY6Params(HexRobotSimParams):
    """Parameters for simulated Archer Y6 (no hardware concepts)."""
    grip_type: str = "gp80"       # "gp80", "gr100", "empty"


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

class HexRobotSimArcherY6(HexRobotSimBase):
    """Simulation-only Archer Y6 robot.

    Usage::

        params = HexRobotSimArcherY6Params(device="cuda:0", headless=False)
        robot = HexRobotSimArcherY6(params)
        robot.start()

        while robot.is_working():
            robot.set_arm_pos_cmd({"jnt_pos": [0.0, -1.5, 3.0, 0.0, 0.0, 0.0]})
            robot.update()

        robot.stop()
    """

    def __init__(
        self,
        params: HexRobotSimArcherY6Params = HexRobotSimArcherY6Params(),
    ) -> None:
        super().__init__(params)
        # State deques for user polling
        self._deque_user: dict[str, deque] = {
            "arm_state": deque(maxlen=params.state_buffer_size),
            "grip_state": deque(maxlen=params.state_buffer_size),
        }
        self._callbacks = {
            "arm_state": self._deque_user["arm_state"].append,
            "grip_state": self._deque_user["grip_state"].append,
        }
        self._has_grip = params.grip_type != "empty"

        # Set by init_vars / init_robot
        self._grip_type: str = "empty"
        self._dof_dict: dict[str, int] = {}
        self._deque_dict: dict[str, Optional[deque]] = {}
        self._cur_cmd: dict[str, Optional[Any]] = {}
        self._cur_state: dict[str, Optional[dict]] = {}

        # IK (lazy — set up by init_robot if EE commands are used)
        self._ik_cfg: Optional[dict] = None
        self._ik_controller = None
        self._ee_body_id: int = -1                  # cached EE body index

        # Stiffness/damping for effort estimation
        self._arm_k: np.ndarray = _ARM_STIFFNESS.copy()
        self._arm_d: np.ndarray = _ARM_DAMPING.copy()
        self._last_arm_target: Optional[np.ndarray] = None

        # Grip joint indices within the full 8-DOF articulation
        self._grip_joint_slice: Optional[list[int]] = None

        # Default joint pos (home)
        self._default_joint_pos: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # init_vars
    # ------------------------------------------------------------------

    def init_vars(self) -> None:
        p = self._params
        self._headless = bool(p.headless)
        self._device = str(p.device)
        self._num_envs = int(p.num_envs)
        self._ctrl_rate = int(p.ctrl_rate)
        self._state_buffer_size = int(p.state_buffer_size)
        self._grip_type = str(p.grip_type)

        arm_dof = 6
        grip_dof = _SIM_GRIP_DOF if self._has_grip else 0
        self._dof_dict = {"arm": arm_dof, "grip": grip_dof}

        self._deque_dict = {
            "arm_cmd": deque(maxlen=self._state_buffer_size),
        }
        self._cur_cmd = {"arm_cmd": None, "grip_cmd": None}

        self._cur_state = {
            "arm": {
                "jnt_pos": np.zeros(arm_dof),
                "jnt_vel": np.zeros(arm_dof),
                "comp_tau": np.zeros(arm_dof),
            },
        }
        if self._has_grip:
            self._deque_dict["grip_cmd"] = deque(maxlen=self._state_buffer_size)
            self._cur_state["grip"] = {
                "jnt_pos": np.zeros(grip_dof),
                "jnt_vel": np.zeros(grip_dof),
                "comp_tau": np.zeros(grip_dof),
            }

    # ------------------------------------------------------------------
    # init_robot
    # ------------------------------------------------------------------

    def init_robot(self) -> None:
        # 1. Create Isaac Lab interface — *first* to satisfy AppLauncher
        from ..sim.isaaclab_interface import IsaacLabSimInterface

        cli_args = ["--headless"] if self._headless else []
        sim = IsaacLabSimInterface()
        sim.initialize(cli_args=cli_args, device=self._device, num_envs=self._num_envs)
        self._sim_interface = sim

        # 2. Now safe to import articulation configs (AppLauncher active)
        cfg = _import_articulation_cfg(self._grip_type)

        # 3. Spawn robot (builds scene)
        sim.spawn_robot("arm_y6", cfg)

        # 4. Step once to populate articulation data buffers
        sim.step()

        # 4. Resolve IK metadata (joint IDs, EE body)
        self._ik_cfg = sim.resolve_arm_ik_cfg("arm_y6")
        self._ee_body_id = self._ik_cfg["ee_body_id"]

        # 5. Create DifferentialIKController for EE control
        from isaaclab.controllers import (
            DifferentialIKController,
            DifferentialIKControllerCfg,
        )
        self._ik_controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=self._num_envs,
            device=self._device,
        )

        # 6. Resolve grip joint indices in the articulation
        if self._has_grip:
            from isaaclab.managers import SceneEntityCfg
            cfg = SceneEntityCfg("arm_y6", joint_names=["J[12]"])
            cfg.resolve(sim._scene)
            self._grip_joint_slice = list(cfg.joint_ids)

        # 7. Store default joint positions (home)
        art = sim._scene["arm_y6"]
        self._default_joint_pos = torch_to_numpy(art.data.default_joint_pos[0])

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Single simulation step: process commands → step → fire callbacks."""
        sim = self._sim_interface
        if sim is None or not sim.is_running():
            return

        # 1. Read current joint state from sim
        arm_pos = sim.get_joint_positions("arm_y6")
        arm_vel = sim.get_joint_velocities("arm_y6")

        self._cur_state["arm"]["jnt_pos"][:] = arm_pos
        self._cur_state["arm"]["jnt_vel"][:] = arm_vel

        if self._has_grip:
            grip_pos_all = sim.get_joint_positions("arm_y6")[self._grip_joint_slice]
            grip_vel_all = sim.get_joint_velocities("arm_y6")[self._grip_joint_slice]
            self._cur_state["grip"]["jnt_pos"][:] = grip_pos_all
            self._cur_state["grip"]["jnt_vel"][:] = grip_vel_all

        # 2. Process arm command
        self._process_arm_cmd()

        # 3. Process grip command
        if self._has_grip:
            self._process_grip_cmd()

        # 4. Step simulation
        sim.step()

        # 5. Build state dataclass and fire callbacks
        try:
            self._fire_arm_state()
            if self._has_grip:
                self._fire_grip_state()
        except Exception as e:
            pass  # state callback failure should not crash the sim loop

    # ------------------------------------------------------------------
    # Command setters — same signatures as real HexRobotArcherY6
    # ------------------------------------------------------------------

    def set_arm_mit_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Set arm direct-impedance (MIT) command.

        In simulation, only ``jnt_pos`` is used (ImplicitActuator PD).
        ``mit_kp`` / ``mit_kd`` are informational.

        Args:
            cmd_dict: keys — ts_ns, jnt_pos, jnt_vel, mit_tau, mit_kp, mit_kd, grav
        """
        ts_ns = cmd_dict.get("ts_ns", ns_now())
        cmd = HexDcRoboArmCtrlStamped(
            header=build_header(ts_ns),
            arm_ctrl=HexDcRoboArmCtrl(
                ctrl_mode=HexDcRoboArmCtrlMode.MIT,
                grav=cmd_dict.get("grav"),
                jnt=build_hex_jnt(
                    cmd_dict.get("jnt_pos"),
                    cmd_dict.get("jnt_vel"),
                    cmd_dict.get("mit_tau"),
                    cmd_dict.get("mit_kp"),
                    cmd_dict.get("mit_kd"),
                    dof=self._dof_dict["arm"],
                ),
            ),
        )
        self._deque_dict["arm_cmd"].append(cmd)

    def set_arm_pos_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Set arm joint-position (JNT) command.

        Args:
            cmd_dict: keys — ts_ns, jnt_pos, jnt_eff, lim_vel, lim_acc, grav
        """
        ts_ns = cmd_dict.get("ts_ns", ns_now())
        dof = self._dof_dict["arm"]
        cmd = HexDcRoboArmCtrlStamped(
            header=build_header(ts_ns),
            arm_ctrl=HexDcRoboArmCtrl(
                ctrl_mode=HexDcRoboArmCtrlMode.JNT,
                grav=cmd_dict.get("grav"),
                jnt=build_hex_jnt(
                    pos=np.asarray(cmd_dict["jnt_pos"]),
                    eff=np.asarray(cmd_dict.get("jnt_eff", np.zeros(dof))),
                    dof=dof,
                ),
            ),
        )
        self._deque_dict["arm_cmd"].append(cmd)

    def set_arm_pose_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Set arm end-effector pose (EE) command — uses Isaac Lab IK.

        Args:
            cmd_dict: keys — ts_ns, pose_pos [x,y,z], pose_quat [w,x,y,z],
                      jnt_eff, lim_vel, lim_acc, grav
        """
        ts_ns = cmd_dict.get("ts_ns", ns_now())
        pose_pos = np.asarray(cmd_dict["pose_pos"])
        pose_quat = np.asarray(cmd_dict.get("pose_quat", [1.0, 0.0, 0.0, 0.0]))
        dof = self._dof_dict["arm"]
        cmd = HexDcRoboArmCtrlStamped(
            header=build_header(ts_ns),
            arm_ctrl=HexDcRoboArmCtrl(
                ctrl_mode=HexDcRoboArmCtrlMode.EE,
                grav=cmd_dict.get("grav"),
                jnt=build_hex_jnt(
                    eff=np.asarray(cmd_dict.get("jnt_eff", np.zeros(dof))),
                    dof=dof,
                ),
                pose=HexDcBasePose(
                    position=HexDcBaseVector3(
                        x=float(pose_pos[0]), y=float(pose_pos[1]), z=float(pose_pos[2]),
                    ),
                    orientation=HexDcBaseQuaternion(
                        w=float(pose_quat[0]), x=float(pose_quat[1]),
                        y=float(pose_quat[2]), z=float(pose_quat[3]),
                    ),
                ),
            ),
        )
        self._deque_dict["arm_cmd"].append(cmd)

    def set_grip_mit_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Set grip direct-impedance (MIT) command.

        In simulation, only ``jnt_pos`` drives the PD target.
        """
        if not self._has_grip:
            return
        ts_ns = cmd_dict.get("ts_ns", ns_now())
        cmd = HexDcRoboGripCtrlStamped(
            header=build_header(ts_ns),
            grip_ctrl=HexDcRoboGripCtrl(
                ctrl_mode=HexDcRoboGripCtrlMode.MIT,
                jnt=build_hex_jnt(
                    cmd_dict.get("jnt_pos"),
                    cmd_dict.get("jnt_vel"),
                    cmd_dict.get("mit_tau"),
                    cmd_dict.get("mit_kp"),
                    cmd_dict.get("mit_kd"),
                    dof=1,  # user-facing DOF
                ),
            ),
        )
        self._deque_dict["grip_cmd"].append(cmd)

    def set_grip_pos_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Set grip joint-position (JNT) command.

        Args:
            cmd_dict: keys — ts_ns, jnt_pos, jnt_eff (max torque), lim_vel
        """
        if not self._has_grip:
            return
        ts_ns = cmd_dict.get("ts_ns", ns_now())
        cmd = HexDcRoboGripCtrlStamped(
            header=build_header(ts_ns),
            grip_ctrl=HexDcRoboGripCtrl(
                ctrl_mode=HexDcRoboGripCtrlMode.JNT,
                jnt=build_hex_jnt(
                    pos=np.atleast_1d(np.asarray(cmd_dict.get("jnt_pos", [0.0]))),
                    eff=np.atleast_1d(np.asarray(cmd_dict.get("jnt_eff", 3.0))),
                    dof=1,
                ),
            ),
        )
        self._deque_dict["grip_cmd"].append(cmd)

    def set_grip_force_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Set grip torque (TAU) command.

        Args:
            cmd_dict: keys — ts_ns, jnt_eff (target torque), lim_vel
        """
        if not self._has_grip:
            return
        ts_ns = cmd_dict.get("ts_ns", ns_now())
        cmd = HexDcRoboGripCtrlStamped(
            header=build_header(ts_ns),
            grip_ctrl=HexDcRoboGripCtrl(
                ctrl_mode=HexDcRoboGripCtrlMode.TAU,
                jnt=build_hex_jnt(
                    eff=np.atleast_1d(np.asarray(cmd_dict["jnt_eff"])),
                    dof=1,
                ),
            ),
        )
        self._deque_dict["grip_cmd"].append(cmd)

    # ------------------------------------------------------------------
    # State getters — same signatures as real HexRobotArcherY6
    # ------------------------------------------------------------------

    def get_arm_state(self, latest: bool = True) -> Optional[HexDcRoboArmStateStamped]:
        """Return latest arm state dataclass, or None."""
        return deque_helper(self._deque_user["arm_state"], latest=latest)

    def get_grip_state(self, latest: bool = True) -> Optional[HexDcRoboGripStateStamped]:
        """Return latest grip state dataclass, or None."""
        if not self._has_grip:
            return None
        return deque_helper(self._deque_user["grip_state"], latest=latest)

    def get_arm_motor_status(self) -> Optional[dict]:
        """Return dict with motor temp/error info (sim: defaults)."""
        return {
            "motor_temp": [0.0] * self._dof_dict["arm"],
            "driver_temp": [0.0] * self._dof_dict["arm"],
            "error": [0] * self._dof_dict["arm"],
        }

    def get_grip_motor_status(self) -> Optional[dict]:
        """Return grip motor status (sim: defaults), or None if grip disabled."""
        if not self._has_grip:
            return None
        return {
            "motor_temp": [0.0],
            "driver_temp": [0.0],
            "error": [0],
        }

    def get_arm_robot_mode(self) -> str:
        """Sim always reports RmRunning."""
        return "RmRunning"

    def is_control_able(self) -> bool:
        """Sim is always controllable."""
        return True

    def get_dofs(self) -> dict[str, int]:
        """Return dict with 'arm' and 'grip' DOF counts."""
        return dict(self._dof_dict)

    # ------------------------------------------------------------------
    # Internal — command processing
    # ------------------------------------------------------------------

    def _process_arm_cmd(self) -> None:
        """Peek latest arm command, apply to sim interface."""
        temp = deque_helper(self._deque_dict["arm_cmd"], latest=True)
        if temp is not None:
            self._cur_cmd["arm_cmd"] = temp

        cmd_stamped = self._cur_cmd["arm_cmd"]
        if cmd_stamped is None:
            return

        arm_ctrl = cmd_stamped.arm_ctrl
        mode = arm_ctrl.ctrl_mode
        jnt_info = arm_ctrl.jnt
        sim = self._sim_interface
        dof = self._dof_dict["arm"]

        if mode == HexDcRoboArmCtrlMode.MIT:
            # Position target only (ImplicitActuator uses PD internally)
            target = np.zeros(dof)
            if jnt_info.pos is not None and jnt_info.pos.size == dof:
                target[:] = np.asarray(jnt_info.pos)
            self._last_arm_target = target
            sim.set_joint_position_target("arm_y6", target)

        elif mode == HexDcRoboArmCtrlMode.JNT:
            target = np.zeros(dof)
            if jnt_info.pos is not None and jnt_info.pos.size == dof:
                target[:] = np.asarray(jnt_info.pos)
            self._last_arm_target = target
            sim.set_joint_position_target("arm_y6", target)

        elif mode == HexDcRoboArmCtrlMode.EE:
            pose = arm_ctrl.pose
            target_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            target_quat = np.array([
                pose.orientation.w, pose.orientation.x,
                pose.orientation.y, pose.orientation.z,
            ])
            joint_target = sim.compute_ik(
                "arm_y6", target_pos, target_quat,
                self._ik_cfg, self._ik_controller,
            )
            self._last_arm_target = joint_target.copy()

            # Merge IK result into full joint target
            full = sim.get_joint_positions("arm_y6").copy()
            for i, ji in enumerate(self._ik_cfg["joint_ids"]):
                full[ji] = joint_target[i]
            sim.set_joint_position_target("arm_y6", full)

        else:
            pass  # unsupported mode — skip

    def _process_grip_cmd(self) -> None:
        """Peek latest grip command, apply to sim interface."""
        temp = deque_helper(self._deque_dict["grip_cmd"], latest=True)
        if temp is not None:
            self._cur_cmd["grip_cmd"] = temp

        cmd_stamped = self._cur_cmd["grip_cmd"]
        if cmd_stamped is None:
            return

        grip_ctrl = cmd_stamped.grip_ctrl
        mode = grip_ctrl.ctrl_mode
        jnt_info = grip_ctrl.jnt
        sim = self._sim_interface

        # User command is 1-DOF; replicate to both J1 / J2
        def _grip_val(arr, idx=0, default=0.0) -> float:
            if arr is not None and arr.size > idx:
                return float(np.asarray(arr).flat[idx])
            return default

        if mode == HexDcRoboGripCtrlMode.MIT:
            pos_v = _grip_val(jnt_info.pos, 0, 0.0)
            target = np.full(_SIM_GRIP_DOF, pos_v)
            sim.set_joint_position_target("arm_y6", target, joint_ids=self._grip_joint_slice)

        elif mode == HexDcRoboGripCtrlMode.JNT:
            pos_v = _grip_val(jnt_info.pos, 0, 0.0)
            target = np.full(_SIM_GRIP_DOF, pos_v)
            sim.set_joint_position_target("arm_y6", target, joint_ids=self._grip_joint_slice)

        elif mode == HexDcRoboGripCtrlMode.TAU:
            eff_v = _grip_val(jnt_info.eff, 0, 0.0)
            target = np.full(_SIM_GRIP_DOF, eff_v)
            sim.set_joint_effort_target("arm_y6", target, joint_ids=self._grip_joint_slice)

    # ------------------------------------------------------------------
    # Internal — state callback firing
    # ------------------------------------------------------------------

    def _fire_arm_state(self) -> None:
        sim = self._sim_interface
        state = self._cur_state["arm"]
        pos = state["jnt_pos"].copy()
        vel = state["jnt_vel"].copy()

        # Estimate effort from PD law: τ = k*(q_d - q) - d*v
        if self._last_arm_target is not None:
            eff = self._arm_k * (self._last_arm_target - pos) - self._arm_d * vel
        else:
            eff = np.zeros(self._dof_dict["arm"])

        # EE pose from sim FK (use cached body ID, avoid per-step resolve)
        ee_pos, ee_quat = sim.get_body_pose_world_by_id("arm_y6", self._ee_body_id)

        state_msg = HexDcRoboArmStateStamped(
            header=build_header(),
            arm_state=HexDcRoboArmState(
                jnt=HexDcBaseJntState(position=pos, velocity=vel, effort=eff),
                pose=HexDcBasePose(
                    position=HexDcBaseVector3(
                        x=float(ee_pos[0]), y=float(ee_pos[1]), z=float(ee_pos[2]),
                    ),
                    orientation=HexDcBaseQuaternion(
                        w=float(ee_quat[0]), x=float(ee_quat[1]),
                        y=float(ee_quat[2]), z=float(ee_quat[3]),
                    ),
                ),
            ),
        )
        self._callbacks["arm_state"](state_msg)

    def _fire_grip_state(self) -> None:
        state = self._cur_state["grip"]
        pos = state["jnt_pos"].copy()[:1]    # expose as 1-DOF
        vel = state["jnt_vel"].copy()[:1]
        eff = np.zeros(1)

        state_msg = HexDcRoboGripStateStamped(
            header=build_header(),
            grip_state=HexDcRoboGripState(
                jnt=HexDcBaseJntState(position=pos, velocity=vel, effort=eff),
            ),
        )
        self._callbacks["grip_state"](state_msg)


# ---------------------------------------------------------------------------
# Module-level helpers  (deferred import to respect AppLauncher constraint)
# ---------------------------------------------------------------------------

def _import_articulation_cfg(grip_type: str):
    """Import the right ArticulationCfg based on grip type.

    This function is called from init_robot(), *after* AppLauncher is created,
    so isaaclab imports are safe.
    """
    key = _GRIP_TO_USD.get(grip_type, "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG")
    from hex_isaac_usd.configs import (
        HEX_ISAAC_USD_ARCHER_Y6_CFG,
        HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG,
    )
    mapping = {
        "HEX_ISAAC_USD_ARCHER_Y6_CFG": HEX_ISAAC_USD_ARCHER_Y6_CFG,
        "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG": HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG,
    }
    return mapping[key]
