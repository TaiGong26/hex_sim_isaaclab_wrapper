"""HexRobotSimArcherY6 — simulation-only Archer Y6 robot.

API aligned with ``hex_driver_robot.robot_archer_y6.HexRobotArcherY6``
(but only simulation-relevant methods, no hardware).

V1 scope: 6-DOF arm + optional GR100 gripper (8-DOF USD).
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from hex_util_ros import HexDynUtilY6, interp_joint

from hex_util_msg.dataclass import (
    HexDcBaseJntState,
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
from hex_util_runtime import HexRate, deque_helper, ns_now

from ..sim.interface import ActuatorCmd
from ..utils import build_header, build_hex_jnt, build_pose, torch_to_numpy
from .base import HexRobotSimBase, HexRobotSimParams


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# USD config aliases — values match key names in hex_isaac_usd.configs
_GRIP_TO_USD: dict[str, str] = {
    "empty": "HEX_ISAAC_USD_ARCHER_Y6_CFG",
    "gp80": "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG",
    "gr100": "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG",
    "gp100": "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG",
}

# JNT/EE position-interpolation default max joint velocity [rad/s], used when
# the command omits ``lim_vel`` (mirrors hex_driver_robot's default).
_ARM_LIM_VEL_DEFAULT = 5.0

# Default arm URDF for EE-mode analytic IK (HexDynUtilY6).  Same gripper-
# compensated model as the Mujoco reference launch.  Resolved from the repo
# layout (editable install keeps ``__file__`` inside the workspace).
_ARM_URDF_PATH_DEFAULT = str(
    Path(__file__).resolve().parents[4]
    / "pkg_ttg/test/urdf/hex_ros_urdf_archer_y6/urdf/gr100_comp.urdf"
)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass
class HexRobotSimArcherY6Params(HexRobotSimParams):
    """Parameters for simulated Archer Y6 (no hardware concepts)."""
    grip_type: str = "gp80"       # "gp80", "gr100", "empty"
    # Arm URDF used for EE-mode analytic IK (HexDynUtilY6).
    urdf_path: str = _ARM_URDF_PATH_DEFAULT


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
            time.sleep(1.0 / params.ctrl_rate)

        robot.stop()
    """

    def __init__(
        self,
        params: HexRobotSimArcherY6Params = HexRobotSimArcherY6Params(),
    ) -> None:
        super().__init__(params=params, name="Archer_y6")
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

        # Actuator names — resolved from articulation at spawn time
        self._arm_actuator: str = ""
        self._grip_actuator: Optional[str] = None

        # Cached EE body index (resolved in init_robot)
        self._ee_body_id: int = -1

        # Default joint pos (home)
        self._default_joint_pos: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # init_vars
    # ------------------------------------------------------------------

    def init_vars(self) -> None:
        """Non-DOF-dependent init.

        DOF counts and the state/PD buffers sized by them are resolved in
        init_robot(), after the USD config is loaded — dof_dict comes from the
        config's actuators. ``_cur_state`` stays ``{}`` until then.
        """
        self._deque_dict = {
            "arm_cmd": deque(maxlen=self._params.state_buffer_size),
        }
        self._cur_cmd = {"arm_cmd": None, "grip_cmd": None}

        # EE-mode analytic IK (same config as the Mujoco reference).
        self._dyn_util = HexDynUtilY6(
            model_path=self._params.urdf_path,
            last_link="link_6",
            pose_end_in_flange=np.array([0.187, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )

        if self._has_grip:
            self._deque_dict["grip_cmd"] = deque(maxlen=self._params.state_buffer_size)

    # ------------------------------------------------------------------
    # init_robot
    # ------------------------------------------------------------------

    def init_robot(self) -> None:
        # 1. Create Isaac Lab interface — *first* to satisfy AppLauncher
        from ..sim.isaaclab_interface import IsaacLabSimInterface

        cli_args = ["--headless"] if bool(self._params.headless) else []
        sim = IsaacLabSimInterface()
        sim.initialize(cli_args=cli_args, device=self._params.device,
                        num_envs=self._params.num_envs, sim_env=self._params.sim_env)
        self._sim_interface = sim

        # 2. Now safe to import articulation configs (AppLauncher active)
        cfg = _import_articulation_cfg(self._params.grip_type)

        # 3. Spawn robot (builds scene)
        sim.spawn_robot("archer_y6", cfg)

        # 4. Step once to populate articulation data buffers
        sim.step()

        # 5. Resolve actuator names from the spawned articulation
        articulation = sim._scene["archer_y6"]
        act_names = list(articulation.actuators.keys())
        self._arm_actuator = act_names[0]
        self._grip_actuator = act_names[1] if len(act_names) > 1 else None
        self.logi(f"Actuators: arm={self._arm_actuator}, grip={self._grip_actuator}")

        # 5b. Resolve DOF counts from the spawned articulation (config-driven).
        arm_act = articulation.actuators[self._arm_actuator]
        self._dof_dict["arm"] = int(arm_act.num_joints)
        self._dof_dict["grip"] = (
            int(articulation.actuators[self._grip_actuator].num_joints)
            if self._grip_actuator is not None else 0
        )
        self.logi(f"DOF from config: arm={self._dof_dict['arm']}, grip={self._dof_dict['grip']}")

        # 5c. Allocate state / PD buffers sized by the resolved DOF.
        dof = self._dof_dict
        self._cur_state = {
            "arm": {
                "jnt_pos": np.zeros(dof["arm"]),
                "jnt_vel": np.zeros(dof["arm"]),
                "jnt_eff": np.zeros(dof["arm"]),
                "comp_tau": np.zeros(dof["arm"]),  # gravity + Coriolis comp [Nm]
            },
        }
        self._arm_kp_default = np.zeros(dof["arm"])
        self._arm_kd_default = np.zeros(dof["arm"])
        self._arm_kp_default_phys = np.zeros(dof["arm"])
        self._arm_kd_default_phys = np.zeros(dof["arm"])
        if self._has_grip:
            self._cur_state["grip"] = {
                "jnt_pos": np.zeros(dof["grip"]),
                "jnt_vel": np.zeros(dof["grip"]),
                "jnt_eff": np.zeros(dof["grip"]),
            }

        # Set Default Kp Kd
        act = articulation.actuators[self._arm_actuator]
        self._arm_kp_default = torch_to_numpy(act.stiffness[0]).copy()
        self._arm_kd_default = torch_to_numpy(act.damping[0]).copy()
        jids = act.joint_indices
        self._arm_kp_default_phys = torch_to_numpy(
            articulation.data.default_joint_stiffness[0, jids]).copy()
        self._arm_kd_default_phys = torch_to_numpy(
            articulation.data.default_joint_damping[0, jids]).copy()
        self.logi(
            f"Arm default PD (motor model): kp={self._arm_kp_default.tolist()}, "
            f"kd={self._arm_kd_default.tolist()}")

        # 6. Resolve EE body index for state FK
        from isaaclab.managers import SceneEntityCfg
        ee_cfg = SceneEntityCfg("archer_y6", body_names=["link_6"])
        ee_cfg.resolve(sim._scene)
        self._ee_body_id = ee_cfg.body_ids[0]

        # 7. Store default joint positions (home)
        self._default_joint_pos = torch_to_numpy(articulation.data.default_joint_pos[0])

    # ------------------------------------------------------------------
    # work_loop — background thread body (heartbeat only, no sim ops)
    # ------------------------------------------------------------------

    def work_loop(self) -> None:
        """Background heartbeat.

        Sim stepping is done synchronously via :meth:`step` on the main
        thread to keep ``SimulationContext.step()`` on the event-loop thread.
        """
        rate = HexRate(self._params.ctrl_rate)
        while self.is_working():
            rate.sleep()

    # ------------------------------------------------------------------
    # step — synchronous sim pipeline (call from main thread)
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Process commands → step simulation → publish state."""
        # 1. Process arm command
        self._process_arm_cmd()
        # 2. Process grip command
        if self._has_grip:
            self._process_grip_cmd()
        # 3. Step simulation (applies pending commands internally)
        self._sim_interface.step()
        # 4. Read state and publish callbacks
        try:
            self._update_arm_state()
            if self._has_grip:
                self._update_grip_state()
        except Exception:
            self.loge("State callback failure", exc_info=True)

    # ------------------------------------------------------------------
    # Command setters — same signatures as real HexRobotArcherY6
    # ------------------------------------------------------------------

    def set_arm_mit_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Set arm direct-impedance (MIT) command.

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
                    pos=cmd_dict.get("jnt_pos"),
                    vel=cmd_dict.get("jnt_vel"),
                    eff=cmd_dict.get("mit_tau"),
                    kp=cmd_dict.get("mit_kp"),
                    kd=cmd_dict.get("mit_kd"),
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
                    lim_vel=np.asarray(cmd_dict.get(
                        "lim_vel", np.full(dof, _ARM_LIM_VEL_DEFAULT))),
                    lim_acc=np.asarray(cmd_dict.get("lim_acc", np.zeros(dof))),
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
                    lim_vel=np.asarray(cmd_dict.get(
                        "lim_vel", np.full(dof, _ARM_LIM_VEL_DEFAULT))),
                    lim_acc=np.asarray(cmd_dict.get("lim_acc", np.zeros(dof))),
                    dof=dof,
                ),
                pose=build_pose(pose_pos, pose_quat),
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
                    pos=cmd_dict.get("jnt_pos"),
                    vel=cmd_dict.get("jnt_vel"),
                    eff=cmd_dict.get("mit_tau"),
                    kp=cmd_dict.get("mit_kp"),
                    kd=cmd_dict.get("mit_kd"),
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

    def get_dofs(self) -> dict[str, int]:
        """Return dict with 'arm' and 'grip' DOF counts."""
        return dict(self._dof_dict)

    # ------------------------------------------------------------------
    # Internal — command processing (work thread context)
    # ------------------------------------------------------------------

    def _process_arm_cmd(self) -> None:
        """Convert latest arm command to ActuatorCmd and push to sim.

        All three modes push a **full MIT-form** command; only the position
        target and PD source differ:

        - MIT: user pos/kp/kd/vel passed through unchanged (no interpolation).
        - JNT/EE: the commanded position is first ``interp_joint``-interpolated
          toward the target and the **load-time default PD** is restored.

        Gravity/Coriolis compensation is **ADDed** to effort in every mode
        (mirrors mujoco's ``+ __cur_comp`` in all three branches).
        """
        temp = deque_helper(self._deque_dict["arm_cmd"], latest=True)
        if temp is not None:
            self._cur_cmd["arm_cmd"] = temp

        cmd_stamped = self._cur_cmd["arm_cmd"]
        if cmd_stamped is None:
            return

        arm_ctrl = cmd_stamped.arm_ctrl
        mode = arm_ctrl.ctrl_mode
        jnt_info = arm_ctrl.jnt
        dof = self._dof_dict["arm"]

        ac = ActuatorCmd()

        # ---- common: compensation ADDed to effort in all modes ----
        comp = self._cur_state["arm"]["comp_tau"]
        eff = (np.asarray(jnt_info.eff, dtype=np.float32)
               if jnt_info.eff is not None and jnt_info.eff.size == dof
               else np.zeros(dof, dtype=np.float32))
        ac.effort = eff + comp

        if mode == HexDcRoboArmCtrlMode.MIT:
            
            if jnt_info.pos is not None and jnt_info.pos.size == dof:
                ac.position = np.asarray(jnt_info.pos, dtype=np.float32)
            if jnt_info.kp is not None and jnt_info.kp.size == dof:
                ac.stiffness = np.asarray(jnt_info.kp, dtype=np.float32)
            if jnt_info.kd is not None and jnt_info.kd.size == dof:
                ac.damping = np.asarray(jnt_info.kd, dtype=np.float32)
            if jnt_info.vel is not None and jnt_info.vel.size == dof:
                ac.velocity = np.asarray(jnt_info.vel, dtype=np.float32)

        elif mode in (HexDcRoboArmCtrlMode.JNT, HexDcRoboArmCtrlMode.EE):
            # JNT/EE: commanded position target → interp_joint → default PD.
            if mode == HexDcRoboArmCtrlMode.JNT:
                if jnt_info.pos is None or jnt_info.pos.size != dof:
                    self.logw("JNT command without pos target — skipped")
                    return
                tar_pos = np.asarray(jnt_info.pos, dtype=np.float32)
            elif mode == HexDcRoboArmCtrlMode.EE:
                ik_success, tar_pos = self._ik_target(arm_ctrl)
                if not ik_success:
                    self.logw("EE IK failed — command skipped")
                    return
                tar_pos = np.asarray(tar_pos, dtype=np.float32)

            cur_pos = self._cur_state["arm"]["jnt_pos"]
            lim_vel = (np.asarray(jnt_info.lim_vel, dtype=np.float32)
                       if jnt_info.lim_vel is not None
                       and jnt_info.lim_vel.size == dof
                       else np.full(dof, _ARM_LIM_VEL_DEFAULT, dtype=np.float32))
            err_limit = lim_vel * (1.0 / self._params.ctrl_rate)
            
            ac.position = interp_joint(cur_pos, tar_pos, err_limit).astype(np.float32)
            ac.stiffness = self._arm_kp_default.astype(np.float32)
            ac.damping = self._arm_kd_default.astype(np.float32)
            ac.velocity = np.zeros(dof, dtype=np.float32)  # mujoco JNT vel term = 0

        # Push only if at least one field was set
        if any(v is not None for v in
               [ac.position, ac.velocity, ac.effort, ac.stiffness, ac.damping]):
            self._sim_interface.push_command(actuator=self._arm_actuator, cmd=ac)

    def _ik_target(self, arm_ctrl) -> tuple[bool, np.ndarray]:
        """Run analytic IK on an EE pose command → (success, target positions).

        Same call as the Mujoco reference (``HexDynUtilY6``, wxyz quaternion).
        """
        pose = arm_ctrl.pose
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        ori = np.array([pose.orientation.w, pose.orientation.x,
                        pose.orientation.y, pose.orientation.z])
        cur_pos = self._cur_state["arm"]["jnt_pos"]
        return self._dyn_util.inverse_kinematics_analytic((pos, ori), cur_pos)

    def _process_grip_cmd(self) -> None:
        """Convert latest grip command to ActuatorCmd and push to sim."""
        temp = deque_helper(self._deque_dict["grip_cmd"], latest=True)
        if temp is not None:
            self._cur_cmd["grip_cmd"] = temp

        cmd_stamped = self._cur_cmd["grip_cmd"]
        if cmd_stamped is None:
            return

        grip_ctrl = cmd_stamped.grip_ctrl
        mode = grip_ctrl.ctrl_mode
        jnt_info = grip_ctrl.jnt
        assert self._grip_actuator is not None

        # User command is 1-DOF; replicate to all grip joints (count from the
        # USD config, read from dof_dict["grip"]).
        if mode in (HexDcRoboGripCtrlMode.MIT, HexDcRoboGripCtrlMode.JNT):
            pos_v = _grip_val(jnt_info.pos, 0, 0.0)
            ac = ActuatorCmd(
                position=np.full(self._dof_dict["grip"], pos_v, dtype=np.float32))
            self._sim_interface.push_command(actuator=self._grip_actuator, cmd=ac)

        elif mode == HexDcRoboGripCtrlMode.TAU:
            eff_v = _grip_val(jnt_info.eff, 0, 0.0)
            ac = ActuatorCmd(
                effort=np.full(self._dof_dict["grip"], eff_v, dtype=np.float32))
            self._sim_interface.push_command(actuator=self._grip_actuator, cmd=ac)

    # ------------------------------------------------------------------
    # Internal — state update and publish (work thread context)
    # ------------------------------------------------------------------

    def _update_arm_state(self) -> None:
        """Read arm joint state from sim → build msg → push to callback deque."""
        sim = self._sim_interface

        pos = sim.get_joint_positions(self._arm_actuator)
        vel = sim.get_joint_velocities(self._arm_actuator)
        eff = sim.get_joint_efforts(self._arm_actuator)

        self._cur_state["arm"]["jnt_pos"][:] = pos
        self._cur_state["arm"]["jnt_vel"][:] = vel
        self._cur_state["arm"]["jnt_eff"][:] = eff

        # Gravity/Coriolis compensation for the current state (one-step delay,
        # same as mujoco — computed from the state just read, applied next step).
        self._cur_state["arm"]["comp_tau"][:] = (
            sim.get_gravity_coriolis_compensation(self._arm_actuator))

        # EE pose from sim FK (use cached body ID)
        ee_pos, ee_quat = sim.get_body_pose_world_by_id(self._ee_body_id)

        state_msg = HexDcRoboArmStateStamped(
            header=build_header(),
            arm_state=HexDcRoboArmState(
                jnt=HexDcBaseJntState(position=pos.copy(), velocity=vel.copy(), effort=eff),
                pose=build_pose(ee_pos, ee_quat),
            ),
        )
        self._callbacks["arm_state"](state_msg)

    def _update_grip_state(self) -> None:
        """Read grip joint state from sim → build msg → push to callback deque."""
        sim = self._sim_interface
        assert self._grip_actuator is not None

        pos = sim.get_joint_positions(self._grip_actuator)
        vel = sim.get_joint_velocities(self._grip_actuator)
        eff = sim.get_joint_efforts(self._grip_actuator)
    
        self._cur_state["grip"]["jnt_pos"][:] = pos
        self._cur_state["grip"]["jnt_vel"][:] = vel
        self._cur_state["grip"]["jnt_eff"][:] = eff

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

def _grip_val(arr, idx: int = 0, default: float = 0.0) -> float:
    """Safely extract a single float from an optional array (1-DOF grip cmd)."""
    if arr is not None and arr.size > idx:
        return float(np.asarray(arr).flat[idx])
    return default


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
