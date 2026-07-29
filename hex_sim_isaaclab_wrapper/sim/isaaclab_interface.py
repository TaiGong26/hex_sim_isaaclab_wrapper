"""IsaacLabSimInterface — Isaac Lab backend for SimInterface.

CRITICAL TIMING CONSTRAINT
--------------------------
``AppLauncher`` MUST be instantiated **before** any ``import isaaclab.*``
statement.  Therefore **all** isaaclab imports in this file happen inside
method bodies, never at module level.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils import numpy_to_torch, torch_to_numpy
from .interface import SimInterface


class IsaacLabSimInterface(SimInterface):
    """Isaac Lab implementation of SimInterface.

    Usage (inside a method, not at module level)::

        sim = IsaacLabSimInterface()
        sim.initialize(cli_args=["--headless"], device="cuda:0", num_envs=1)
        sim.spawn_robot("arm", some_articulation_cfg)
        # Ready to step:
        sim.step()
        pos = sim.get_joint_positions("arm")
    """

    def __init__(self):
        self._app = None
        self._sim = None
        self._scene = None
        self._num_envs = 1
        self._sim_dt = 0.0
        self._device = "cuda:0"
        self._robot_configs: dict[str, tuple] = {}
        self._scene_ready = False
        self._num_joints: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, cli_args: Optional[list[str]] = None,
                   device: str = "cuda:0", num_envs: int = 1,
                   camera_pos: tuple[float, float, float] = (2.5, 0.0, 4.0),
                   camera_target: tuple[float, float, float] = (0.0, 0.0, 2.0)) -> None:
        """Boot Isaac Lab.

        Must be called first — creates AppLauncher + SimulationContext.
        """
        # ---- critical: AppLauncher before ANY isaaclab import ----
        import argparse
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(description="IsaacLabSimInterface")
        AppLauncher.add_app_launcher_args(parser)
        parsed, _ = parser.parse_known_args(cli_args)
        parsed.device = device

        app_launcher = AppLauncher(parsed)
        self._app = app_launcher.app
        self._num_envs = num_envs
        self._device = device

        # ---- now safe to import isaaclab modules ----
        import isaaclab.sim as sim_utils

        sim_cfg = sim_utils.SimulationCfg(device=device)
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

    def spawn_robot(self, name: str, articulation_cfg,
                    prim_path: Optional[str] = None) -> None:
        """Register a robot and build the scene immediately."""
        if self._scene_ready:
            raise RuntimeError("Scene already built. Multi-robot not yet supported.")
        if prim_path is None:
            prim_path = "{ENV_REGEX_NS}/" + name
        self._robot_configs[name] = (articulation_cfg, prim_path)
        self._build_scene()

    def get_num_joints(self, name: str) -> int:
        """Return number of joints for a previously spawned robot."""
        if name in self._num_joints:
            return self._num_joints[name]
        raise RuntimeError(f"Joint count for '{name}' not available yet.")

    # ------------------------------------------------------------------
    # Internal accessor — reduces boilerplate in getter/setter methods
    # ------------------------------------------------------------------

    def _get_articulation(self, name: str):
        """Return the Articulation object, raising if scene isn't ready."""
        self._require_scene_ready()
        return self._scene[name]

    def _require_scene_ready(self) -> None:
        """Raise RuntimeError if the scene has not been built yet."""
        if not self._scene_ready or self._scene is None:
            raise RuntimeError(
                "Scene not ready. Call step() at least once after spawn_robot()."
            )

    # ------------------------------------------------------------------
    # State reading  → numpy
    # ------------------------------------------------------------------

    def get_joint_positions(self, name: str) -> np.ndarray:
        articulation = self._get_articulation(name)
        return torch_to_numpy(articulation.data.joint_pos[0])

    def get_joint_velocities(self, name: str) -> np.ndarray:
        articulation = self._get_articulation(name)
        return torch_to_numpy(articulation.data.joint_vel[0])

    def get_joint_efforts(self, name: str) -> np.ndarray:
        articulation = self._get_articulation(name)
        return torch_to_numpy(articulation.data.applied_forces[0])

    def get_body_pose(self, name: str, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in the base (root) frame."""
        articulation = self._get_articulation(name)
        from isaaclab.managers import SceneEntityCfg
        import isaaclab.utils.math as math_utils

        cfg = SceneEntityCfg(name, body_names=[body_name])
        cfg.resolve(self._scene)
        body_idx = cfg.body_ids[0]
        body_pose_w = articulation.data.body_pose_w[0, body_idx]
        root_pose_w = articulation.data.root_pose_w[0]

        pos_b, quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:3].unsqueeze(0),
            root_pose_w[3:7].unsqueeze(0),
            body_pose_w[:3].unsqueeze(0),
            body_pose_w[3:7].unsqueeze(0),
        )
        return torch_to_numpy(pos_b[0]), torch_to_numpy(quat_b[0])

    def get_body_pose_world(self, name: str, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in the world frame."""
        self._require_scene_ready()
        from isaaclab.managers import SceneEntityCfg
        cfg = SceneEntityCfg(name, body_names=[body_name])
        cfg.resolve(self._scene)
        body_idx = cfg.body_ids[0]
        return self.get_body_pose_world_by_id(name, body_idx)

    def get_body_pose_world_by_id(self, name: str, body_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in world frame by pre-resolved body index."""
        articulation = self._get_articulation(name)
        body_pose_w = articulation.data.body_pose_w[0, body_idx]
        return torch_to_numpy(body_pose_w[:3]), torch_to_numpy(body_pose_w[3:7])

    def get_root_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        articulation = self._get_articulation(name)
        root_pose = articulation.data.root_pose_w[0]
        return torch_to_numpy(root_pose[:3]), torch_to_numpy(root_pose[3:7])

    # ------------------------------------------------------------------
    # Command writing  ← numpy
    # ------------------------------------------------------------------

    def set_joint_position_target(self, name: str, target: np.ndarray,
                                   joint_ids: Optional[list[int]] = None) -> None:
        articulation = self._get_articulation(name)
        t = numpy_to_torch(target, self._device).unsqueeze(0)
        if joint_ids is not None:
            full = articulation.data.joint_pos[0].clone()
            full[joint_ids] = t[0]
            articulation.set_joint_position_target(full.unsqueeze(0))
        else:
            articulation.set_joint_position_target(t)

    def set_joint_velocity_target(self, name: str, target: np.ndarray,
                                   joint_ids: Optional[list[int]] = None) -> None:
        articulation = self._get_articulation(name)
        t = numpy_to_torch(target, self._device).unsqueeze(0)
        if joint_ids is not None:
            full = articulation.data.joint_vel[0].clone()
            full[joint_ids] = t[0]
            articulation.set_joint_velocity_target(full.unsqueeze(0))
        else:
            articulation.set_joint_velocity_target(t)

    def set_joint_effort_target(self, name: str, target: np.ndarray,
                                 joint_ids: Optional[list[int]] = None) -> None:
        articulation = self._get_articulation(name)
        t = numpy_to_torch(target, self._device).unsqueeze(0)
        if joint_ids is not None:
            full = articulation.data.applied_forces[0].clone()
            full[joint_ids] = t[0]
            articulation.set_joint_effort_target(full.unsqueeze(0))
        else:
            articulation.set_joint_effort_target(t)

    def set_joint_state(self, name: str, position: np.ndarray, velocity: np.ndarray) -> None:
        articulation = self._get_articulation(name)
        p = numpy_to_torch(position, self._device).unsqueeze(0)
        v = numpy_to_torch(velocity, self._device).unsqueeze(0)
        articulation.write_joint_state_to_sim(p, v)
        articulation.reset()

    def get_sim_dt(self) -> float:
        return self._sim_dt

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance physics by one dt."""
        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self._sim_dt)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _build_scene(self) -> None:
        """Dynamically build the InteractiveScene from registered configs."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass

        scene_attrs = {
            "ground": AssetBaseCfg(
                prim_path="/World/defaultGroundPlane",
                spawn=sim_utils.GroundPlaneCfg(),
            ),
            "dome_light": AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            ),
        }
        for robot_name, (cfg, prim_path) in self._robot_configs.items():
            scene_attrs[robot_name] = cfg.replace(prim_path=prim_path)

        # Dynamically create a scene config class with the registered robots
        SceneCfg = configclass(
            type("DynamicSceneCfg", (InteractiveSceneCfg,), scene_attrs)
        )
        scene_cfg = SceneCfg(num_envs=self._num_envs, env_spacing=2.0)
        self._scene = InteractiveScene(scene_cfg)
        self._sim.reset()
        self._scene_ready = True

        # Cache joint counts
        for robot_name in self._robot_configs:
            articulation = self._scene[robot_name]
            self._num_joints[robot_name] = torch_to_numpy(
                articulation.data.joint_pos[0]
            ).shape[0]
