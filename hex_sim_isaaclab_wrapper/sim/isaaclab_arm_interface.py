"""IsaacLabArmInterface — Isaac Lab backend for SimInterface.

## Critical timing constraint

`AppLauncher` MUST be instantiated **before** any `import isaaclab.*`
statement. Therefore **all** isaaclab imports in this file happen inside
method bodies, never at module level.

## Thread-safety note

`ArticulationData` is **not** thread-safe (no locks, no clones on read).
In this design all sim data access (getters, step, apply_commands) happens on
a single work thread. The `_pending_cmds` dict is also work-thread-only.
Cross-thread boundary is at the robot layer's deques (`_deque_dict` /
`_deque_user`).

## Import-race avoidance

`initialize()` eagerly loads **all** isaaclab modules that other methods
will need, so they are in `sys.modules` before the work thread ever starts.
Method-level import statements are then just trivial cache lookups — no
race condition even when called on a background thread.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from isaaclab.assets import ArticulationCfg
    from isaaclab.assets.articulation.articulation import Articulation

from ..utils import numpy_to_torch, torch_to_numpy
from .interface import SimInterface, ActuatorCmd


class IsaacLabArmInterface(SimInterface):
    """Isaac Lab implementation of `SimInterface`.

    Example (inside a method, not at module level):

    ```python
    sim = IsaacLabArmInterface()
    sim.initialize(cli_args=["--headless"], device="cuda:0", num_envs=1)
    sim.spawn_robot("arm", some_articulation_cfg)
    sim.step()  # ready to step
    pos = sim.get_joint_positions("archer_y6")
    ```
    """

    def __init__(self) -> None:
        """Initialize interface state; call `initialize()` before use."""
        self._app = None
        self._sim = None
        self._scene = None
        self._num_envs = 1
        self._sim_dt = 0.0
        self._render_interval = 1
        self._sim_step_counter = 0
        self._device = "cuda:0"
        self._robot_configs: dict[str, tuple] = {}
        self._scene_ready = False
        self._sim_env = None
        # Single-robot support: name stored here, getters/setters don't need it
        self._robot_name: str | None = None
        # Pending commands: work-thread-only, applied in step()
        self._pending_cmds: dict[str, ActuatorCmd] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, cli_args: Optional[list[str]] = None,
                   device: str = "cuda:0", num_envs: int = 1, dt: float = 60.0,
                   render_rate: float = 60.0,
                   camera_pos: tuple[float, float, float] = (2.5, 0.0, 4.0),
                   camera_target: tuple[float, float, float] = (0.0, 0.0, 2.0),
                   sim_env=None) -> None:
        """Boot Isaac Lab.

        Must be called first — creates AppLauncher + SimulationContext.

        Args:
            cli_args:     CLI argument list (e.g. `["--headless"]`).
            device:       Torch device string.
            num_envs:     Number of parallel environments.
            dt:           Control-loop rate [Hz]; sim step dt = 1/dt.
            render_rate:  Rendering frequency [Hz]; physics steps per render
                          = round(dt / render_rate).
            camera_pos:   Initial camera position.
            camera_target: Initial camera look-at target.
            sim_env:      Environment config (a `HexSimEnvParams`-like object
                          with `ground_cfg`, `dome_light_cfg`, etc.).
                          `None` uses built-in defaults.
        """
        # ---- critical: AppLauncher before ANY isaaclab import ----
        import argparse
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="IsaacLabArmInterface")
        AppLauncher.add_app_launcher_args(parser)
        parsed, _ = parser.parse_known_args(cli_args)
        parsed.device = device

        app_launcher = AppLauncher(parsed)
        self._app = app_launcher.app
        self._num_envs = num_envs
        self._device = device
        self._sim_env = sim_env

        # ---- now safe to import isaaclab modules ----
        import isaaclab.sim as sim_utils

        # Eager-import every isaaclab module that other methods will need.
        # This puts them in sys.modules now (on the main thread), so the
        # same import statements in work-thread methods are no-ops.
        import isaaclab.utils.math as math_utils  # noqa: F401
        from isaaclab.assets import AssetBaseCfg  # noqa: F811
        from isaaclab.managers import SceneEntityCfg  # noqa: F811
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: F811
        from isaaclab.utils import configclass  # noqa: F811

        render_interval = max(1, int(round(dt / render_rate)))
        self._render_interval = render_interval
        sim_cfg = sim_utils.SimulationCfg(device=device, dt=(1/dt),
                                          render_interval=render_interval)
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._sim_dt = self._sim.get_physics_dt()
        self._sim.set_camera_view(camera_pos, camera_target)
        self._scene = None
        self._scene_ready = False

    def close(self) -> None:
        """Shut down the simulation app."""
        if self._app is not None:
            self._app.close()
            self._app = None
        self._sim = None
        self._scene = None
        self._scene_ready = False

    def is_running(self) -> bool:
        """Return True while the app is not being asked to exit."""
        return self._app is not None and self._app.is_running()

    # ------------------------------------------------------------------
    # Robot spawning
    # ------------------------------------------------------------------

    def spawn_robot(self, name: str, articulation_cfg: ArticulationCfg,
                    prim_path: Optional[str] = None) -> None:
        """Register a robot and build the scene immediately."""
        if self._scene_ready:
            raise RuntimeError("Scene already built. Multi-robot not yet supported.")
        if prim_path is None:
            prim_path = "{ENV_REGEX_NS}/" + name
        self._robot_name = name
        self._robot_configs[name] = (articulation_cfg, prim_path)
        self._build_scene()

    def get_num_joints(self) -> int:
        """Return number of joints for the spawned robot."""
        articulation = self._get_articulation()
        return articulation.data.joint_pos.shape[-1]

    # ------------------------------------------------------------------
    # Internal accessor
    # ------------------------------------------------------------------

    def _get_articulation(self) -> Articulation:
        """Return the `Articulation` object, raising if the scene isn't ready.

        Uses the internally stored `_robot_name` — no caller-side name needed.
        """
        self._require_scene_ready()
        return self._scene[self._robot_name]

    def _require_scene_ready(self) -> None:
        """Raise RuntimeError if the scene has not been built yet."""
        if not self._scene_ready or self._scene is None:
            raise RuntimeError(
                "Scene not ready. Call step() at least once after spawn_robot()."
            )

    # ------------------------------------------------------------------
    # State reading  → numpy
    # ------------------------------------------------------------------

    def get_joint_positions(self, actuator: str) -> np.ndarray:
        """Read joint positions [rad] for an actuator as a `(num_joints,)` array."""
        articulation = self._get_articulation()
        jids = articulation.actuators[actuator].joint_indices
        joint_pos = articulation.data.joint_pos.clone()
        return torch_to_numpy(joint_pos[0, jids])

    def get_joint_velocities(self, actuator: str) -> np.ndarray:
        """Read joint velocities [rad/s] for an actuator as a `(num_joints,)` array."""
        articulation = self._get_articulation()
        jids = articulation.actuators[actuator].joint_indices
        joint_vel = articulation.data.joint_vel.clone()
        return torch_to_numpy(joint_vel[0, jids])

    def get_joint_efforts(self, actuator: str) -> np.ndarray:
        """Read joint efforts [Nm] for an actuator as a `(num_joints,)` array.

        > **Note:** For `ImplicitActuator`, the returned torque is an
        *approximation* computed by Isaac Lab (PhysX does not expose implicit
        PD torque). The `applied_torque` tensor is updated in-place during
        each step.
        """
        articulation = self._get_articulation()
        jids = articulation.actuators[actuator].joint_indices
        torque = articulation.data.applied_torque.clone()
        return torch_to_numpy(torque[0, jids])

    def get_gravity_coriolis_compensation(self, actuator: str) -> np.ndarray:
        """Read gravity + Coriolis/centrifugal compensation torques [Nm].

        PhysX-native: gravity compensation + Coriolis/centrifugal compensation,
        summed = `C*dq + G` for the current articulation state.
        """
        articulation = self._get_articulation()
        jids = articulation.actuators[actuator].joint_indices
        gravity = articulation.root_physx_view.get_gravity_compensation_forces()
        coriolis = articulation.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
        return torch_to_numpy(gravity[0, jids] + coriolis[0, jids])

    def _resolve_body_idx(self, body_name: str) -> int:
        """Resolve a body name to its index in the articulation's body array."""
        self._require_scene_ready()
        from isaaclab.managers import SceneEntityCfg

        cfg = SceneEntityCfg(self._robot_name, body_names=[body_name])
        cfg.resolve(self._scene)
        return cfg.body_ids[0]

    def _body_pose_world(self, body_idx: int) -> torch.Tensor:
        """World pose `[pos(3), quat_wxyz(4)]` of a body, shape `(7,)`."""
        articulation = self._get_articulation()
        return articulation.data.body_pose_w[0, body_idx]

    def _relative_pose(self, base_pose_w: torch.Tensor,
                       target_pose_w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Express `target_pose_w` relative to `base_pose_w` → (pos, quat wxyz)."""
        import isaaclab.utils.math as math_utils

        pos_b, quat_b = math_utils.subtract_frame_transforms(
            base_pose_w[:3].unsqueeze(0),
            base_pose_w[3:7].unsqueeze(0),
            target_pose_w[:3].unsqueeze(0),
            target_pose_w[3:7].unsqueeze(0),
        )
        return torch_to_numpy(pos_b[0]), torch_to_numpy(quat_b[0])

    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in the base (root) frame."""
        body_idx = self._resolve_body_idx(body_name)
        articulation = self._get_articulation()
        return self._relative_pose(articulation.data.root_pose_w[0],
                                   self._body_pose_world(body_idx))

    def get_body_pose_world(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in the world frame."""
        return self.get_body_pose_world_by_id(self._resolve_body_idx(body_name))

    def get_body_pose_world_by_id(self, body_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in world frame by pre-resolved body index."""
        body_pose_w = self._body_pose_world(body_idx)
        return torch_to_numpy(body_pose_w[:3]), torch_to_numpy(body_pose_w[3:7])

    def get_body_pose_relative_by_ids(self, base_body_idx: int,
                                      target_body_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Read target body pose relative to the base body frame (base → target)."""
        return self._relative_pose(self._body_pose_world(base_body_idx),
                                   self._body_pose_world(target_body_idx))

    def get_root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the root (base) pose: (position [m], quaternion [wxyz])."""
        articulation = self._get_articulation()
        root_pose = articulation.data.root_pose_w[0]
        return torch_to_numpy(root_pose[:3]), torch_to_numpy(root_pose[3:7])

    
    # ------------------------------------------------------------------
    # Command writing  ← numpy
    # ------------------------------------------------------------------

    def push_command(self, actuator: str, cmd: ActuatorCmd) -> None:
        """Push a command for an actuator. Overwrites previous pending cmd."""
        self._pending_cmds[actuator] = cmd

    def _apply_commands(self) -> None:
        """Apply all pending commands to the articulation.

        Called from `step()` — operates on the work thread only.
        """
        if not self._pending_cmds:
            return
        articulation = self._get_articulation()
        for actuator_name, cmd in self._pending_cmds.items():
            _articulation = articulation.actuators[actuator_name]
            jids = _articulation.joint_indices

            # Position target
            if cmd.position is not None:
                t = numpy_to_torch(cmd.position, self._device).unsqueeze(0)
                articulation.set_joint_position_target(t, joint_ids=jids)

            # Velocity target
            if cmd.velocity is not None:
                t = numpy_to_torch(cmd.velocity, self._device).unsqueeze(0)
                articulation.set_joint_velocity_target(t, joint_ids=jids)

            # Effort target
            if cmd.effort is not None:
                t = numpy_to_torch(cmd.effort, self._device).unsqueeze(0)
                articulation.set_joint_effort_target(t, joint_ids=jids)

            # Stiffness / damping — dual path (motor model + PhysX)
            if cmd.stiffness is not None or cmd.damping is not None:
                kp = (numpy_to_torch(cmd.stiffness, self._device)
                      if cmd.stiffness is not None
                      else _articulation.stiffness.clone())
                kd = (numpy_to_torch(cmd.damping, self._device)
                      if cmd.damping is not None
                      else _articulation.damping.clone())
                _articulation.stiffness[:] = kp
                _articulation.damping[:] = kd
                articulation.write_joint_stiffness_to_sim(kp, joint_ids=jids)
                articulation.write_joint_damping_to_sim(kd, joint_ids=jids)

        # self._pending_cmds.clear()

    def set_joint_state(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Set joint positions/velocities directly, then reset the articulation.

        Args:
            position: Target joint positions [rad].
            velocity: Target joint velocities [rad/s].
        """
        articulation = self._get_articulation()
        p = numpy_to_torch(position, self._device).unsqueeze(0)
        v = numpy_to_torch(velocity, self._device).unsqueeze(0)
        articulation.write_joint_state_to_sim(p, v)
        articulation.reset()

    def get_sim_time(self) -> float:
        """Return the current simulation time [s]."""
        return self._sim.current_time

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance physics by one dt; render every `_render_interval` steps (GUI only).

        Order: apply pending commands → flush to sim → physics step → throttled
        render → update scene. Mirror of `ManagerBasedEnv`'s render loop.
        """
        self._apply_commands()
        self._scene.write_data_to_sim()
        self._sim_step_counter += 1
        self._sim.step(render=False)
        if self._sim_step_counter % self._render_interval == 0 and self._sim.has_gui():
            self._sim.render()
        self._scene.update(self._sim_dt)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _build_scene(self) -> None:
        """Dynamically build the InteractiveScene from registered configs.

        Scene composition is driven by `self._sim_env` (set via
        `initialize()`). If `_sim_env` is `None`, built-in defaults
        (ground plane + dome light) are used.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass

        env = self._sim_env
        scene_attrs: dict = {}

        # Ground plane — custom or default
        if env is not None and env.ground_cfg is not None:
            scene_attrs["ground"] = AssetBaseCfg(
                prim_path=env.ground_prim_path, spawn=env.ground_cfg)
        else:
            scene_attrs["ground"] = AssetBaseCfg(
                prim_path="/World/defaultGroundPlane",
                spawn=sim_utils.GroundPlaneCfg())

        # Dome light — custom or default
        if env is not None and env.dome_light_cfg is not None:
            scene_attrs["dome_light"] = AssetBaseCfg(
                prim_path=env.dome_light_prim_path, spawn=env.dome_light_cfg)
        else:
            scene_attrs["dome_light"] = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=3000.0, color=(0.75, 0.75, 0.75)))

        # Additional user-requested assets
        if env is not None:
            for asset_name, asset_cfg in env.additional_assets.items():
                scene_attrs[asset_name] = asset_cfg

        # Registered robots
        env_spacing = env.env_spacing if env is not None else 2.0
        for robot_name, (cfg, prim_path) in self._robot_configs.items():
            scene_attrs[robot_name] = cfg.replace(prim_path=prim_path)

        # Dynamically create a scene config class
        SceneCfg = configclass(
            type("DynamicSceneCfg", (InteractiveSceneCfg,), scene_attrs)
        )
        scene_cfg = SceneCfg(num_envs=self._num_envs, env_spacing=env_spacing)
        self._scene = InteractiveScene(scene_cfg)
        self._sim.reset()
        self._scene_ready = True
