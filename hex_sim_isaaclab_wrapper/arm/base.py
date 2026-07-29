"""Base classes for the sim robot wrapper.

Model
-----
- ``HexRobotSimParams`` — pure simulation parameters (no hardware concepts).
- ``HexRobotSimBase`` — abstract base, synchronous (no thread).  Users call
  ``start()`` → ``update()`` in a loop → ``stop()``.

Key differences from real ``HexRobotBase``:

* No ``_work_thread`` / ``_stop_event`` — simulation steps are synchronous.
* ``start()`` only initialises (non-blocking).  The caller drives the loop.
* ``is_working()`` delegates to ``SimInterface.is_running()``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class HexRobotSimParams:
    """Simulation parameters (no hardware concepts like host/port).

    Attributes:
        ctrl_rate:        Simulation control loop rate [Hz].
        state_buffer_size: Number of state deque entries to retain.
        num_envs:          Number of parallel environments (Isaac Lab).
        device:            Torch device string (e.g. ``"cuda:0"``, ``"cpu"``).
        headless:          Run without rendering window.
    """
    ctrl_rate: float = 500.0
    state_buffer_size: int = 100
    num_envs: int = 1
    device: str = "cuda:0"
    headless: bool = False

    # Additional CLI args forwarded to Isaac Lab's AppLauncher.
    # This allows users to pass custom flags without touching our API.
    extra_cli_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base robot
# ---------------------------------------------------------------------------

class HexRobotSimBase(ABC):
    """Abstract sim robot — synchronous lifecycle, no daemon thread.

    Usage::

        robot = HexRobotSimArcherY6(params)
        robot.start()
        while robot.is_working():
            time.sleep(1.0 / params.ctrl_rate)
            robot.set_arm_pos_cmd({...})
            robot.update()
        robot.stop()
    """

    def __init__(self, params: HexRobotSimParams) -> None:
        self._params = params
        self._sim_interface = None  # set by subclass in init_robot()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize internal vars and the sim interface (incl. scene)."""
        self.init_vars()
        self.init_robot()

    def stop(self) -> None:
        """Shut down the simulation interface."""
        if self._sim_interface is not None:
            self._sim_interface.close()
            self._sim_interface = None

    def is_working(self) -> bool:
        """Return True while the simulation app is still running."""
        if self._sim_interface is None:
            return False
        return self._sim_interface.is_running()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def init_vars(self) -> None:
        """Initialize internal data structures (params → member variables)."""
        ...

    @abstractmethod
    def init_robot(self) -> None:
        """Create and initialise the SimInterface, spawn robot(s) in scene."""
        ...

    @abstractmethod
    def update(self) -> None:
        """Single simulation step:

        1. Read fresh state from sim.
        2. Dispatch user commands to sim interface.
        3. Step the physics.
        4. Fire state callbacks.
        """
        ...
