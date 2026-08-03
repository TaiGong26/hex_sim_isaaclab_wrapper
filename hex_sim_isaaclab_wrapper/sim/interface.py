"""SimInterface — simulator-agnostic abstraction.

The interface exchanges data via **numpy** arrays (not torch) so that
switching backends (Isaac Lab, Mujoco, …) never leaks a framework dependency.

Concrete implementation: ``IsaacLabSimInterface`` in ``sim.isaaclab_interface``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Unified command dataclass — replaces 4 separate setter methods
# ---------------------------------------------------------------------------


@dataclass
class ActuatorCmd:
    """Unified actuator command — all fields optional.

    ``None`` means "do not modify this quantity" when the command is applied
    in :meth:`SimInterface.step`.

    - ``position`` + ``stiffness``/``damping`` → ImplicitActuator PD target
    - ``effort`` → feed-forward torque override
    - ``velocity`` → velocity target
    """
    position:   Optional[np.ndarray] = None   # joint position target [rad]
    velocity:   Optional[np.ndarray] = None   # joint velocity target [rad/s]
    effort:     Optional[np.ndarray] = None   # feed-forward torque [Nm]
    stiffness:  Optional[np.ndarray] = None   # PD stiffness (kp)
    damping:    Optional[np.ndarray] = None   # PD damping (kd)


# ---------------------------------------------------------------------------
# Simulator abstraction
# ---------------------------------------------------------------------------


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
    def spawn_robot(self, name: str, articulation_cfg,
                    prim_path: Optional[str] = None) -> None:
        """Register and spawn a robot articulation in the scene.

        The *articulation_cfg* is an ``isaaclab.assets.ArticulationCfg``
        (or any backend-specific config); the interface knows how to consume it.

        After spawning, subsequent getter/setter calls use internally
        stored robot name — the *name* parameter is not repeated.

        Args:
            name:             Entity name used to reference the robot later.
            articulation_cfg: Backend-specific articulation configuration.
            prim_path:        USD prim path (e.g. ``"{ENV_REGEX_NS}/MyRobot"``).
                              If None, a default path is generated.
        """
        ...

    # ------------------------------------------------------------------
    # State reading  (numpy out)  —  actuator required, no name
    # ------------------------------------------------------------------

    @abstractmethod
    def get_num_joints(self) -> int:
        """Return number of joints for the spawned robot."""
        ...

    @abstractmethod
    def get_joint_positions(self, actuator: str) -> np.ndarray:
        """Read joint positions [rad] as (num_joints,) numpy array.

        Args:
            actuator: Actuator group name (key in ``ArticulationCfg.actuators``).
        """
        ...

    @abstractmethod
    def get_joint_velocities(self, actuator: str) -> np.ndarray:
        """Read joint velocities [rad/s] as (num_joints,) numpy array.

        Args:
            actuator: Actuator group name.
        """
        ...

    @abstractmethod
    def get_joint_efforts(self, actuator: str) -> np.ndarray:
        """Read joint efforts [Nm] as (num_joints,) numpy array.

        Args:
            actuator: Actuator group name.
        """
        ...

    @abstractmethod
    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read a rigid-body pose in the base frame: (position, quaternion [wxyz]).

        Both arrays are shape (3,) and (4,) respectively.
        """
        ...

    @abstractmethod
    def get_root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the root (base) pose: (position [m], quaternion [wxyz])."""
        ...

    @abstractmethod
    def get_body_pose_world(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in world frame: (position [m], quaternion [wxyz])."""
        ...

    @abstractmethod
    def get_body_pose_world_by_id(self, body_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Read body pose in world frame by pre-resolved body index."""
        ...

    # ------------------------------------------------------------------
    # Command writing  (numpy in)  —  unified via ActuatorCmd
    # ------------------------------------------------------------------

    @abstractmethod
    def push_command(self, actuator: str, cmd: ActuatorCmd) -> None:
        """Push a command for an actuator.

        The command is applied during the next :meth:`step()` call.
        If a previous command for the same actuator is pending, it is
        overwritten (latest wins).

        Args:
            actuator: Actuator group name.
            cmd:      Unified command with optional fields.
        """
        ...


    @abstractmethod
    def get_sim_time(self) -> float:
        """Return the physics timestep [s]."""
        ...

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one physics dt.

        Internally: apply pending commands → ``write_data_to_sim()`` →
        ``sim.step()`` → ``scene.update(dt)``.
        """
        ...
