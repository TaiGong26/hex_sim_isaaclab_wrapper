"""SimInterface — simulator-agnostic abstraction.

The interface exchanges data via **numpy** arrays (not torch) so that
switching backends (Isaac Lab, Mujoco, …) never leaks a framework dependency.

Concrete implementation: ``IsaacLabSimInterface`` in ``sim.isaaclab_interface``.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class SimInterface(ABC):
    """Abstract simulator backend — all data in/out via numpy."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, cli_args: Optional[list[str]] = None,
                   device: str = "cuda:0", num_envs: int = 1) -> None:
        """Boot the simulator (AppLauncher, SimulationContext, …).

        Args:
            cli_args: CLI argument list for parsing (e.g. ``["--headless"]``).
                      If None, uses ``sys.argv[1:]``.
            device:   Torch device string.
            num_envs: Number of parallel environments.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Shut down the simulator (close app, release resources)."""
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Return True while the simulation app should keep running."""
        ...

    # ------------------------------------------------------------------
    # Robot spawning
    # ------------------------------------------------------------------

    @abstractmethod
    def spawn_robot(self, name: str, articulation_cfg, prim_path: Optional[str] = None) -> None:
        """Register and spawn a robot articulation in the scene.

        The *articulation_cfg* is an ``isaaclab.assets.ArticulationCfg``
        (or any backend-specific config); the interface knows how to consume it.

        Args:
            name:          Entity name used to reference the robot later.
            articulation_cfg: Backend-specific articulation configuration.
            prim_path:     USD prim path (e.g. ``"{ENV_REGEX_NS}/MyRobot"``).
                           If None, a default path is generated.
        """
        ...

    @abstractmethod
    def get_num_joints(self, name: str) -> int:
        """Return number of joints for a previously spawned robot."""
        ...

    # ------------------------------------------------------------------
    # State reading  (numpy out)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_joint_positions(self, name: str) -> np.ndarray:
        """Read joint positions [rad] as (num_joints,) numpy array."""
        ...

    @abstractmethod
    def get_joint_velocities(self, name: str) -> np.ndarray:
        """Read joint velocities [rad/s] as (num_joints,) numpy array."""
        ...

    @abstractmethod
    def get_joint_efforts(self, name: str) -> np.ndarray:
        """Read joint efforts [Nm] as (num_joints,) numpy array."""
        ...

    @abstractmethod
    def get_body_pose(self, name: str, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read a rigid-body pose: (position [m], quaternion [wxyz]).

        Both are numpy arrays of shape (3,) and (4,) respectively.
        """
        ...

    @abstractmethod
    def get_root_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read the root (base) pose: (position [m], quaternion [wxyz])."""
        ...

    @abstractmethod
    def get_body_pose_world(self, name: str, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in world frame: (position [m], quaternion [wxyz])."""
        ...

    # ------------------------------------------------------------------
    # Command writing  (numpy in)
    # ------------------------------------------------------------------

    @abstractmethod
    def set_joint_position_target(self, name: str, target: np.ndarray,
                                  joint_ids: Optional[list[int]] = None) -> None:
        """Set PD position target(s).

        If *joint_ids* is given, only those indices are updated.
        """
        ...

    @abstractmethod
    def set_joint_velocity_target(self, name: str, target: np.ndarray,
                                  joint_ids: Optional[list[int]] = None) -> None:
        """Set velocity target(s).

        If *joint_ids* is given, only those indices are updated.
        """
        ...

    @abstractmethod
    def set_joint_effort_target(self, name: str, target: np.ndarray,
                                joint_ids: Optional[list[int]] = None) -> None:
        """Set feed-forward effort target(s).

        If *joint_ids* is given, only those indices are updated.
        """
        ...

    @abstractmethod
    def set_joint_state(self, name: str, position: np.ndarray, velocity: np.ndarray) -> None:
        """Override joint state in simulation (for reset)."""
        ...

    @abstractmethod
    def get_sim_dt(self) -> float:
        """Return the physics timestep [s]."""
        ...

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one physics dt.

        Internally: ``write_data_to_sim() → sim.step() → scene.update(dt)``.
        """
        ...
