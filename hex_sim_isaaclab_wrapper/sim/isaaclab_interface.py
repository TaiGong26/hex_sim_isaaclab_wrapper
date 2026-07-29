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
        self._robot_configs: dict[str, tuple] = {}      # name → (cfg, prim_path)
        self._scene_ready = False

        # Cached joint-count map
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

        Args:
            cli_args:     CLI argument list (e.g. ``["--headless"]``).
            device:       Torch device string.
            num_envs:     Number of parallel environments.
            camera_pos:   Initial camera position (x, y, z).
            camera_target: Initial camera look-at target (x, y, z).
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

        # Set default camera view so user can see the robot
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
        """Register a robot and build the scene immediately.

        For V1, this builds the scene on the first call.  Subsequent
        calls raise (multi-robot support to be added in V2).
        """
        if self._scene_ready:
            raise RuntimeError(
                "Scene already built. Multi-robot not yet supported."
            )
        if prim_path is None:
            prim_path = "{ENV_REGEX_NS}/" + name
        self._robot_configs[name] = (articulation_cfg, prim_path)
        self._build_scene()

    def get_num_joints(self, name: str) -> int:
        """Return number of joints for a previously spawned robot."""
        if name in self._num_joints:
            return self._num_joints[name]
        raise RuntimeError(
            f"Joint count for '{name}' not available yet."
        )

    # ------------------------------------------------------------------
    # State reading  → numpy
    # ------------------------------------------------------------------

    def get_joint_positions(self, name: str) -> np.ndarray:
        _require_scene(self)
        art = self._scene[name]
        return torch_to_numpy(art.data.joint_pos[0])

    def get_joint_velocities(self, name: str) -> np.ndarray:
        _require_scene(self)
        art = self._scene[name]
        return torch_to_numpy(art.data.joint_vel[0])

    def get_joint_efforts(self, name: str) -> np.ndarray:
        _require_scene(self)
        art = self._scene[name]
        # applied efforts from the previous physics step
        return torch_to_numpy(art.data.applied_forces[0])

    def get_body_pose(self, name: str, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in the base (root) frame."""
        _require_scene(self)
        art = self._scene[name]

        from isaaclab.managers import SceneEntityCfg
        import isaaclab.utils.math as math_utils
        import torch

        cfg = SceneEntityCfg(name, body_names=[body_name])
        cfg.resolve(self._scene)
        body_idx = cfg.body_ids[0]
        body_pose_w = art.data.body_pose_w[0, body_idx]  # (7,)
        root_pose_w = art.data.root_pose_w[0]            # (7,)

        pos_b, quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:3].unsqueeze(0),
            root_pose_w[3:7].unsqueeze(0),
            body_pose_w[:3].unsqueeze(0),
            body_pose_w[3:7].unsqueeze(0),
        )
        return torch_to_numpy(pos_b[0]), torch_to_numpy(quat_b[0])

    def get_body_pose_world(self, name: str, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in the world frame."""
        _require_scene(self)
        art = self._scene[name]

        from isaaclab.managers import SceneEntityCfg
        cfg = SceneEntityCfg(name, body_names=[body_name])
        cfg.resolve(self._scene)
        body_idx = cfg.body_ids[0]
        return self.get_body_pose_world_by_id(name, body_idx)

    def get_body_pose_world_by_id(self, name: str, body_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in world frame by pre-resolved body index.

        Faster than ``get_body_pose_world`` when the body ID is known
        (avoids re-resolving every call).
        """
        _require_scene(self)
        art = self._scene[name]
        body_pose_w = art.data.body_pose_w[0, body_idx]
        return torch_to_numpy(body_pose_w[:3]), torch_to_numpy(body_pose_w[3:7])

    def get_root_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        _require_scene(self)
        art = self._scene[name]
        root_pose = art.data.root_pose_w[0]
        return torch_to_numpy(root_pose[:3]), torch_to_numpy(root_pose[3:7])

    # ------------------------------------------------------------------
    # Command writing  ← numpy
    # ------------------------------------------------------------------

    def set_joint_position_target(self, name: str, target: np.ndarray,
                                   joint_ids: Optional[list[int]] = None) -> None:
        _require_scene(self)
        art = self._scene[name]
        t = numpy_to_torch(target, self._device).unsqueeze(0)
        if joint_ids is not None:
            full = art.data.joint_pos[0].clone()
            full[joint_ids] = t[0]
            art.set_joint_position_target(full.unsqueeze(0))
        else:
            art.set_joint_position_target(t)

    def set_joint_velocity_target(self, name: str, target: np.ndarray,
                                   joint_ids: Optional[list[int]] = None) -> None:
        _require_scene(self)
        art = self._scene[name]
        t = numpy_to_torch(target, self._device).unsqueeze(0)
        if joint_ids is not None:
            full = art.data.joint_vel[0].clone()
            full[joint_ids] = t[0]
            art.set_joint_velocity_target(full.unsqueeze(0))
        else:
            art.set_joint_velocity_target(t)

    def set_joint_effort_target(self, name: str, target: np.ndarray,
                                 joint_ids: Optional[list[int]] = None) -> None:
        _require_scene(self)
        art = self._scene[name]
        t = numpy_to_torch(target, self._device).unsqueeze(0)
        if joint_ids is not None:
            full = art.data.applied_forces[0].clone()
            full[joint_ids] = t[0]
            art.set_joint_effort_target(full.unsqueeze(0))
        else:
            art.set_joint_effort_target(t)

    def set_joint_state(self, name: str, position: np.ndarray, velocity: np.ndarray) -> None:
        _require_scene(self)
        art = self._scene[name]
        p = numpy_to_torch(position, self._device).unsqueeze(0)
        v = numpy_to_torch(velocity, self._device).unsqueeze(0)
        art.write_joint_state_to_sim(p, v)
        art.reset()

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

    # ------------------------------------------------------------------
    # IK helpers  (Isaac Lab specific — not part of SimInterface ABC)
    # ------------------------------------------------------------------

    def resolve_arm_ik_cfg(self, name: str) -> dict:
        """Resolve joint IDs for arm joints and EE body for IK.

        Returns dict with keys:
            ``joint_ids`` — list of integer joint indices for ``joint_[1-6]``
            ``ee_body_id`` — integer body index for ``link_6``
            ``ee_jacobi_idx`` — Jacobian index for the EE body
        """
        from isaaclab.managers import SceneEntityCfg
        _require_scene(self)
        art = self._scene[name]
        cfg = SceneEntityCfg(name, joint_names=["joint_[1-6]"], body_names=["link_6"])
        cfg.resolve(self._scene)
        joint_ids = cfg.joint_ids
        ee_body_id = cfg.body_ids[0]
        ee_jacobi_idx = ee_body_id - 1 if art.is_fixed_base else ee_body_id
        return {
            "joint_ids": joint_ids,
            "ee_body_id": ee_body_id,
            "ee_jacobi_idx": ee_jacobi_idx,
        }

    def compute_ik(self, name: str, target_pos: np.ndarray,
                   target_quat: np.ndarray,
                   ik_cfg: dict,
                   ik_controller) -> np.ndarray:
        """Run one IK step, return joint position target as numpy array.

        Args:
            name:   Robot entity name in the scene.
            target_pos:   Target EE position [m], shape (3,).
            target_quat:  Target EE orientation [wxyz], shape (4,).
            ik_cfg:       Dict from ``resolve_arm_ik_cfg()``.
            ik_controller: ``DifferentialIKController`` instance (preserved
                           across calls for damped-least-squares stability).

        Returns:
            Joint position target as (num_arm_joints,) numpy array.
        """
        import torch
        import isaaclab.utils.math as math_utils

        _require_scene(self)
        art = self._scene[name]
        jids = ik_cfg["joint_ids"]
        eid = ik_cfg["ee_body_id"]
        jac = ik_cfg["ee_jacobi_idx"]

        jacobian = art.root_physx_view.get_jacobians()[:, jac, :, jids]
        ee_pose_w = art.data.body_pose_w[:, eid]
        root_pose_w = art.data.root_pose_w
        joint_pos_arm = art.data.joint_pos[:, jids]

        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )

        t_pos = torch.tensor([target_pos], device=self._device)
        t_quat = torch.tensor([target_quat], device=self._device)
        ik_controller.set_command(torch.cat([t_pos, t_quat], dim=-1))

        joint_pos_des = art.data.joint_pos.clone()
        joint_pos_des[:, jids] = ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos_arm
        )
        return torch_to_numpy(joint_pos_des[0, jids])

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
                spawn=sim_utils.DomeLightCfg(
                    intensity=3000.0, color=(0.75, 0.75, 0.75)
                ),
            ),
        }
        for robot_name, (cfg, prim_path) in self._robot_configs.items():
            scene_attrs[robot_name] = cfg.replace(prim_path=prim_path)

        SceneCfg = configclass(
            type("DynamicSceneCfg", (InteractiveSceneCfg,), scene_attrs)
        )
        scene_cfg = SceneCfg(num_envs=self._num_envs, env_spacing=2.0)
        self._scene = InteractiveScene(scene_cfg)
        self._sim.reset()
        self._scene_ready = True

        # Cache joint counts
        for robot_name in self._robot_configs:
            art = self._scene[robot_name]
            self._num_joints[robot_name] = torch_to_numpy(
                art.data.joint_pos[0]
            ).shape[0]


# ---------------------------------------------------------------------------
# Cheap guard to avoid repeating the check everywhere
# ---------------------------------------------------------------------------

def _require_scene(iface: IsaacLabSimInterface) -> None:
    if not iface._scene_ready or iface._scene is None:
        raise RuntimeError(
            "Scene not ready. Call step() at least once "
            "after spawn_robot()."
        )
