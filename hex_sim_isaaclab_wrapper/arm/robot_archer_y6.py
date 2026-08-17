"""HexRobotSimArcherY6 — simulation-only Archer Y6 robot.

V1 scope: 6-DOF arm + optional GR100 gripper (8-DOF USD).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from hex_util_ros import HexDynUtilY6, interp_joint

from hex_util_msg.dataclass import (
    HexDcBaseJntFull,
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
from hex_util_runtime import HexRate, deque_helper

from ..sim.interface import ActuatorCmd
from ..utils import build_header, build_hex_jnt, build_pose, torch_to_numpy
from .base import HexRobotSimBase, HexRobotSimParams

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Grip variant → articulation config key.
_GRIP_TO_USD: dict[str, str] = {
    "empty": "HEX_ISAAC_USD_ARCHER_Y6_CFG",
    "gp80": "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG",
    "gr100": "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG",
    "gp100": "HEX_ISAAC_USD_ARCHER_Y6_GR100_CFG",
}

#: JNT/EE position-interpolation default max joint velocity [rad/s], used when
#: the command omits ``lim_vel``.
_ARM_LIM_VEL_DEFAULT = 5.0

#: Per-arm pose link names (resolved in init_robot). The base link is the arm's
#: first link frame; the EE link is the last link of the arm chain.
_ARM_BASE_LINK = "base_link"
_ARM_EE_LINK = "link_6"


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass
class HexRobotSimArcherY6Params(HexRobotSimParams):
    """Parameters for simulated Archer Y6 (no hardware concepts).

    Attributes:
        grip_type:  Grip variant — `"gp80"`, `"gr100"`, or `"empty"`.
        urdf_path:  URDF for EE-mode analytic IK (`HexDynUtilY6`); `None`
                    disables EE/IK.
    """
    grip_type: str = "gp80"       # "gp80", "gr100", "empty"
    # URDF for EE-mode analytic IK (HexDynUtilY6); None disables EE/IK.
    urdf_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

class HexRobotSimArcherY6(HexRobotSimBase):
    """Simulation-only Archer Y6 robot.

    Example:

    ```python
    params = HexRobotSimArcherY6Params(torch_device="cuda:0", isaac_headless=False)
    robot = HexRobotSimArcherY6(params)
    robot.start()

    while robot.is_working():
        robot.set_arm_pos_cmd({"jnt_pos": [0.0, -1.5, 3.0, 0.0, 0.0, 0.0]})
        time.sleep(1.0 / params.ctrl_rate)

    robot.stop()
    ```
    """

    def __init__(
        self,
        params: HexRobotSimArcherY6Params = HexRobotSimArcherY6Params(),
    ) -> None:
        """Create the Archer Y6 sim robot with default params."""
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
        self._dyn_util: Optional[HexDynUtilY6] = None  # built in init_vars if urdf set

        # Actuator names — resolved from articulation at spawn time
        self._arm_actuator: str = ""
        self._grip_actuator: Optional[str] = None

        # Per-arm body indices (resolved in init_robot): arm name →
        # (base_body_id, ee_body_id). Single-entry for now; dual-arm ready.
        self._arm_refs: dict[str, tuple[int, int]] = {}

        # Default joint pos (home)
        self._default_joint_pos: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # init_vars
    # ------------------------------------------------------------------

    def init_vars(self) -> None:
        """Non-DOF-dependent init.

        DOF counts and the state/PD buffers sized by them are resolved in
        init_robot(), after the USD config is loaded — dof_dict comes from the
        config's actuators. `_cur_state` stays `{}` until then.
        """
        self._deque_dict = {
            "arm_cmd": deque(maxlen=self._params.state_buffer_size),
        }
        self._cur_cmd = {"arm_cmd": None, "grip_cmd": None}

        # EE-mode analytic IK; disabled when no URDF is configured
        # (urdf_path=None → EE commands are skipped).
        if self._params.urdf_path is not None:
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
        """Create the Isaac Lab interface and spawn the robot.

        Initializes the arm interface, spawns `archer_y6` from the selected
        `ArticulationCfg`, then resolves actuator names, DOF counts, default
        PD gains, and per-arm base/EE body indices from the spawned
        articulation. Also allocates the state / PD buffers sized by DOF.
        """
        # 1. Create Isaac Lab interface — *first* to satisfy AppLauncher
        from ..sim.isaaclab_arm_interface import IsaacLabArmInterface

        cli_args = ["--headless"] if bool(self._params.isaac_headless) else []
        sim = IsaacLabArmInterface()
        sim.initialize(cli_args=cli_args, device=self._params.torch_device,
                       dt=self._params.ctrl_rate, num_envs=self._params.sim_num_envs,
                       sim_env=self._params.sim_env,
                       render_rate=self._params.render_rate)
        self._sim_interface = sim

        # 2. Now safe to import articulation configs (AppLauncher active)
        articulation_cfg = _import_articulation_cfg(self._params.grip_type)

        # 3. Spawn robot (builds scene)
        sim.spawn_robot("archer_y6", articulation_cfg)

        # 4. Step once to populate articulation data buffers
        sim.step()

        # 5. Resolve actuator names from the spawned articulation
        articulation = sim._scene["archer_y6"]
        actuator_names = list(articulation.actuators.keys())
        self._arm_actuator = actuator_names[0]
        self._grip_actuator = actuator_names[1] if len(actuator_names) > 1 else None
        self.logi(f"Actuators: arm={self._arm_actuator}, grip={self._grip_actuator}")

        # 5b. Resolve DOF counts from the spawned articulation (config-driven).
        arm_actuator_cfg = articulation.actuators[self._arm_actuator]
        self._dof_dict["arm"] = int(arm_actuator_cfg.num_joints)
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

        # Set default kp/kd.
        arm_actuator_cfg = articulation.actuators[self._arm_actuator]
        self._arm_kp_default = torch_to_numpy(arm_actuator_cfg.stiffness[0]).copy()
        self._arm_kd_default = torch_to_numpy(arm_actuator_cfg.damping[0]).copy()
        joint_indices = arm_actuator_cfg.joint_indices
        self._arm_kp_default_phys = torch_to_numpy(
            articulation.data.default_joint_stiffness[0, joint_indices]).copy()
        self._arm_kd_default_phys = torch_to_numpy(
            articulation.data.default_joint_damping[0, joint_indices]).copy()
        self.logi(
            f"Arm default PD (motor model): kp={self._arm_kp_default.tolist()}, "
            f"kd={self._arm_kd_default.tolist()}")

        # 6. Resolve per-arm base/EE body indices for state FK
        from isaaclab.managers import SceneEntityCfg
        base_body_cfg = SceneEntityCfg("archer_y6", body_names=[_ARM_BASE_LINK])
        base_body_cfg.resolve(sim._scene)
        ee_body_cfg = SceneEntityCfg("archer_y6", body_names=[_ARM_EE_LINK])
        ee_body_cfg.resolve(sim._scene)
        self._arm_refs["arm"] = (base_body_cfg.body_ids[0], ee_body_cfg.body_ids[0])

        # 7. Store default joint positions (home)
        self._default_joint_pos = torch_to_numpy(articulation.data.default_joint_pos[0])

    # ------------------------------------------------------------------
    # work_loop — background thread body (heartbeat only, no sim ops)
    # ------------------------------------------------------------------

    def work_loop(self) -> None:
        """Background heartbeat.

        Sim stepping is done synchronously via `step()` on the main
        thread to keep `SimulationContext.step()` on the event-loop thread.
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
    # Command setters
    # ------------------------------------------------------------------

    def set_arm_mit_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Set arm direct-impedance (MIT) command.

        Args:
            cmd_dict: Keys — `jnt_pos`, `jnt_vel`, `mit_tau`,
                      `mit_kp`, `mit_kd`, `grav`.
        """
        sim_time = self.get_sim_time()
        sim_time = int(sim_time*1e9) if sim_time is not None else None
        cmd = HexDcRoboArmCtrlStamped(
            header=build_header(sim_time),
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
            cmd_dict: Keys — `jnt_pos`, `jnt_eff`, `lim_vel`,
                      `lim_acc`, `grav`.
        """
        sim_time = self.get_sim_time()
        sim_time = int(sim_time*1e9) if sim_time is not None else None
        dof = self._dof_dict["arm"]
        cmd = HexDcRoboArmCtrlStamped(
            header=build_header(sim_time),
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
        """Set arm end-effector pose (EE) command

        Args:
            cmd_dict: Keys — `pose_pos` [x,y,z], `pose_quat` [w,x,y,z],
                      `jnt_eff`, `lim_vel`, `lim_acc`, `grav`.
        """
        sim_time = self.get_sim_time()
        sim_time = int(sim_time*1e9) if sim_time is not None else None
        
        pose_pos = np.asarray(cmd_dict["pose_pos"])
        pose_quat = np.asarray(cmd_dict.get("pose_quat", [1.0, 0.0, 0.0, 0.0]))
        dof = self._dof_dict["arm"]
        cmd = HexDcRoboArmCtrlStamped(
            header=build_header(sim_time),
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

        In simulation, only `jnt_pos` drives the PD target.
        """
        if not self._has_grip:
            return
        sim_time = self.get_sim_time()
        sim_time = int(sim_time*1e9) if sim_time is not None else None
        
        cmd = HexDcRoboGripCtrlStamped(
            header=build_header(sim_time),
            grip_ctrl=HexDcRoboGripCtrl(
                ctrl_mode=HexDcRoboGripCtrlMode.MIT,
                jnt=build_hex_jnt(
                    pos=cmd_dict.get("jnt_pos"),
                    vel=cmd_dict.get("jnt_vel"),
                    eff=cmd_dict.get("mit_tau"),
                    kp=cmd_dict.get("mit_kp"),
                    kd=cmd_dict.get("mit_kd"),
                    dof=self._dof_dict["grip"],  # user-facing DOF
                ),
            ),
        )
        self._deque_dict["grip_cmd"].append(cmd)

    def set_grip_pos_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Set grip joint-position (JNT) command.

        Args:
            cmd_dict: Keys — `jnt_pos`, `jnt_eff` (max torque),
                      `lim_vel`.
        """
        if not self._has_grip:
            return
        sim_time = self.get_sim_time()
        sim_time = int(sim_time*1e9) if sim_time is not None else None
        
        cmd = HexDcRoboGripCtrlStamped(
            header=build_header(sim_time),
            grip_ctrl=HexDcRoboGripCtrl(
                ctrl_mode=HexDcRoboGripCtrlMode.JNT,
                jnt=build_hex_jnt(
                    pos=np.atleast_1d(np.asarray(cmd_dict.get("jnt_pos", [0.0]))),
                    eff=np.atleast_1d(np.asarray(cmd_dict.get("jnt_eff", 3.0))),
                    dof=self._dof_dict["grip"],
                ),
            ),
        )
        self._deque_dict["grip_cmd"].append(cmd)

    def set_grip_force_cmd(self, cmd_dict: dict[str, Any]) -> None:
        """Set grip torque (TAU) command.

        Args:
            cmd_dict: Keys — `jnt_eff` (target torque), `lim_vel`.
        """
        if not self._has_grip:
            return
        sim_time = self.get_sim_time()
        sim_time = int(sim_time*1e9) if sim_time is not None else None
        
        cmd = HexDcRoboGripCtrlStamped(
            header=build_header(sim_time),
            grip_ctrl=HexDcRoboGripCtrl(
                ctrl_mode=HexDcRoboGripCtrlMode.TAU,
                jnt=build_hex_jnt(
                    eff=np.atleast_1d(np.asarray(cmd_dict["jnt_eff"])),
                    dof=self._dof_dict["grip"],
                ),
            ),
        )
        self._deque_dict["grip_cmd"].append(cmd)

    # ------------------------------------------------------------------
    # State getters
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

    def get_arm_ee_pose(self, arm_name: str = "arm") -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return `(pos, quat)` of an arm's EE relative to its baselink, or None.

        `arm_name` defaults to `"arm"` for the single-arm robot; the dict lookup
        is ready for future dual-arm names. Returns `None` if the arm is unknown
        or the sim interface is not yet initialized.
        """
        refs = self._arm_refs.get(arm_name)
        if refs is None or self._sim_interface is None:
            return None
        base_id, ee_id = refs
        return self._sim_interface.get_body_pose_relative_by_ids(base_id, ee_id)

    def get_sim_time(self) -> Optional[float]:
        """Return sim time in seconds (Isaac Lab sim time, not wall-clock)."""
        if self._sim_interface is None:
            return None
        return self._sim_interface.get_sim_time()
    
    # ------------------------------------------------------------------
    # Internal — command processing (work thread context)
    # ------------------------------------------------------------------

    def _process_arm_cmd(self) -> None:
        """Convert latest arm command to ActuatorCmd and push to sim.

        All three modes push a **full MIT-form** command; only the position
        target and PD source differ:

        - MIT: user pos/kp/kd/vel passed through unchanged (no interpolation).
        - JNT: commanded position first `interp_joint`-interpolated toward
          the target, then the **load-time default PD** is restored.
        - EE: pose command resolved by analytic IK (requires `urdf_path`),
          then the same interpolation + default PD as JNT.

        Gravity/Coriolis compensation is **ADDed** to effort in every mode.
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

        _arm_actuator_cmd = ActuatorCmd()

        # ---- common: compensation ADDed to effort in all modes ----
        comp = self._cur_state["arm"]["comp_tau"]
        eff = (np.asarray(jnt_info.eff, dtype=np.float32)
               if jnt_info.eff is not None and jnt_info.eff.size == dof
               else np.zeros(dof, dtype=np.float32))
        _arm_actuator_cmd.effort = eff + comp

        if mode == HexDcRoboArmCtrlMode.MIT:
            if jnt_info.pos is not None and jnt_info.pos.size == dof:
                _arm_actuator_cmd.position = np.asarray(jnt_info.pos, dtype=np.float32)
            if jnt_info.kp is not None and jnt_info.kp.size == dof:
                _arm_actuator_cmd.stiffness = np.asarray(jnt_info.kp, dtype=np.float32)
            if jnt_info.kd is not None and jnt_info.kd.size == dof:
                _arm_actuator_cmd.damping = np.asarray(jnt_info.kd, dtype=np.float32)
            if jnt_info.vel is not None and jnt_info.vel.size == dof:
                _arm_actuator_cmd.velocity = np.asarray(jnt_info.vel, dtype=np.float32)

        elif mode == HexDcRoboArmCtrlMode.JNT:
            if jnt_info.pos is None or jnt_info.pos.size != dof:
                self.logw("JNT command without pos target — skipped")
                return
            self._set_position_interp_command(
                _arm_actuator_cmd, np.asarray(jnt_info.pos, dtype=np.float32), jnt_info, dof)

        elif mode == HexDcRoboArmCtrlMode.EE:
            if self._dyn_util is None:
                self.logw("EE mode requires a URDF (urdf_path=None) — "
                          "cannot solve IK, command skipped")
                return
            ik_success, tar_pos = self._ik_target(arm_ctrl)
            if not ik_success:
                self.logw("EE IK failed — command skipped")
                return
            self._set_position_interp_command(
                _arm_actuator_cmd, np.asarray(tar_pos, dtype=np.float32), jnt_info, dof)

        # Push only if at least one field was set
        if any(v is not None for v in
               [_arm_actuator_cmd.position, _arm_actuator_cmd.velocity, _arm_actuator_cmd.effort, _arm_actuator_cmd.stiffness, _arm_actuator_cmd.damping]):
            self._sim_interface.push_command(actuator=self._arm_actuator, cmd=_arm_actuator_cmd)

    def _ik_target(self, arm_ctrl: HexDcRoboArmCtrl) -> tuple[bool, np.ndarray]:
        """Run analytic IK on an EE pose command → (success, target positions).

        Uses `HexDynUtilY6` with a wxyz quaternion.
        """
        pose = arm_ctrl.pose
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        ori = np.array([pose.orientation.w, pose.orientation.x,
                        pose.orientation.y, pose.orientation.z])
        cur_pos = self._cur_state["arm"]["jnt_pos"]
        return self._dyn_util.inverse_kinematics_analytic((pos, ori), cur_pos)

    def _set_position_interp_command(
        self, actuator_cmd: ActuatorCmd, target_pos: np.ndarray,
        jnt_info: HexDcBaseJntFull, dof: int,
    ) -> None:
        """Interpolate toward `target_pos` and restore load-time default PD.

        Shared by the JNT and EE branches: commanded position is first
        `interp_joint`-limited toward the target, then the load-time default
        stiffness/damping are applied.
        """
        current_pos = self._cur_state["arm"]["jnt_pos"]
        lim_vel = (np.asarray(jnt_info.lim_vel, dtype=np.float32)
                   if jnt_info.lim_vel is not None
                   and jnt_info.lim_vel.size == dof
                   else np.full(dof, _ARM_LIM_VEL_DEFAULT, dtype=np.float32))
        err_limit = lim_vel * (1.0 / self._params.ctrl_rate)

        actuator_cmd.position = interp_joint(current_pos, target_pos, err_limit).astype(np.float32)
        actuator_cmd.stiffness = self._arm_kp_default.astype(np.float32)
        actuator_cmd.damping = self._arm_kd_default.astype(np.float32)
        actuator_cmd.velocity = np.zeros(dof, dtype=np.float32)  # JNT: no velocity term

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
        if mode == HexDcRoboGripCtrlMode.MIT:
            pos_v = _grip_val(jnt_info.pos, 0, 0.0)
            _grip_actuator_cmd = ActuatorCmd(
                position=np.full(self._dof_dict["grip"], pos_v, dtype=np.float32))
            self._sim_interface.push_command(actuator=self._grip_actuator, cmd=_grip_actuator_cmd)

        elif mode == HexDcRoboGripCtrlMode.JNT:
            pos_v = _grip_val(jnt_info.pos, 0, 0.0)
            _grip_actuator_cmd = ActuatorCmd(
                position=np.full(self._dof_dict["grip"], pos_v, dtype=np.float32))
            self._sim_interface.push_command(actuator=self._grip_actuator, cmd=_grip_actuator_cmd)

        elif mode == HexDcRoboGripCtrlMode.TAU:
            eff_v = _grip_val(jnt_info.eff, 0, 0.0)
            _grip_actuator_cmd = ActuatorCmd(
                effort=np.full(self._dof_dict["grip"], eff_v, dtype=np.float32))
            self._sim_interface.push_command(actuator=self._grip_actuator, cmd=_grip_actuator_cmd)

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
        # computed from the state just read, applied next step).
        self._cur_state["arm"]["comp_tau"][:] = (
            sim.get_gravity_coriolis_compensation(self._arm_actuator))
    
        # EE pose relative to the arm's baselink (per-arm pose, dual-arm-ready)
        base_id, ee_id = self._arm_refs["arm"]
        ee_pos, ee_quat = sim.get_body_pose_relative_by_ids(base_id, ee_id)
        
        sim_time = self.get_sim_time()
        sim_time = int(sim_time*1e9) if sim_time is not None else None
        
        state_msg = HexDcRoboArmStateStamped(
            header=build_header(sim_time),
            arm_state=HexDcRoboArmState(
                jnt=HexDcBaseJntState(position=pos.copy(), velocity=vel.copy(), effort=eff.copy()),
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
        
        sim_time = self.get_sim_time()
        sim_time = int(sim_time*1e9) if sim_time is not None else None

        state_msg = HexDcRoboGripStateStamped(
            header=build_header(sim_time),
            grip_state=HexDcRoboGripState(
                jnt=HexDcBaseJntState(position=pos.copy(), velocity=vel.copy(), effort=eff.copy()),
            ),
        )
        self._callbacks["grip_state"](state_msg)

# ---------------------------------------------------------------------------
# Module-level helpers  (deferred import to respect AppLauncher constraint)
# ---------------------------------------------------------------------------

def _grip_val(
    arr: Optional[np.ndarray], idx: int = 0, default: float = 0.0,
) -> float:
    """Safely extract a single float from an optional array (1-DOF grip cmd)."""
    if arr is not None and arr.size > idx:
        return float(np.asarray(arr).flat[idx])
    return default


def _import_articulation_cfg(grip_type: str) -> ArticulationCfg:
    """Import the right `ArticulationCfg` based on the grip type.

    Called from `init_robot()` — *after* AppLauncher is created — so
    isaaclab imports are safe here.
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
