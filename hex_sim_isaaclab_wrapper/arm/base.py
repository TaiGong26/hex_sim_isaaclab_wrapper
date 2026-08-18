"""Base classes for the sim robot wrapper.

## Model

- `HexRobotSimParams` — pure simulation parameters (no hardware concepts).
- `HexRobotSimBase` — abstract base, **main-thread driven**. Users call
  `start()` to initialise the simulator, then poll state via
  `get_arm_state()`. `stop()` closes the simulator.

## Lifecycle

1. `robot = HexRobotSimArcherY6(params)` — constructor.
2. `robot.start()` — calls `init_vars()` + `init_robot()` (synchronous;
   creates the Isaac Lab app on the calling thread).
3. Main thread drives the sim:
   `while robot.is_working(): robot.set_*(); robot.step()`.
4. `robot.stop()` — closes the sim.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from ..tools.log import setup_logger


# ---------------------------------------------------------------------------
# Environment / Scene configuration
# ---------------------------------------------------------------------------

@dataclass
class HexSimEnvParams:
    """Scene environment configuration passed to SimInterface.

    Each field `*_cfg` accepts an Isaac Lab `SpawnCfg` (e.g.
    `GroundPlaneCfg`, `DomeLightCfg`). `None` means the
    SimInterface will use its own built-in default.
    """
    ground_prim_path: str = "/World/defaultGroundPlane"
    ground_cfg: Any = None
    dome_light_prim_path: str = "/World/Light"
    dome_light_cfg: Any = None
    env_spacing: float = 2.0
    additional_assets: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class HexRobotSimParams:
    """Simulation parameters (no hardware concepts like host/port).

    Attributes:
        ctrl_rate:        Simulation control loop rate [Hz].
        render_rate:      Rendering frequency [Hz]; interface converts to
                          `render_interval = round(ctrl_rate / render_rate)`.
        state_buffer_size: Number of state deque entries to retain.
        torch_device:      Torch device string (e.g. `"cuda:0"`, `"cpu"`).
        isaac_headless:    Run without rendering window.
    """
    ctrl_rate: float = 1000.0
    render_rate: float = 60.0
    state_buffer_size: int = 100
    torch_device: str = "cuda:0"
    isaac_headless: bool = False

    # Additional CLI args forwarded to Isaac Lab's AppLauncher.
    # This allows users to pass custom flags without touching our API.
    extra_cli_args: list[str] = field(default_factory=list)

    # Environment/scene configuration (ground, lights, etc.)
    sim_env: HexSimEnvParams = field(default_factory=HexSimEnvParams)


# ---------------------------------------------------------------------------
# Base robot
# ---------------------------------------------------------------------------

class HexRobotSimBase(ABC):
    """Abstract sim robot — main-thread-driven lifecycle.

    Example:

    ```python
    robot = HexRobotSimArcherY6(params)
    robot.start()                     # synchronous init
    while robot.is_working():
        robot.set_arm_pos_cmd({...})
        robot.step()                  # process → sim step → read state
        state = robot.get_arm_state()
        time.sleep(1.0 / params.ctrl_rate)
    robot.stop()                      # closes sim
    ```
    """

    def __init__(self, params: HexRobotSimParams, name: str) -> None:
        """Store params and set up logging."""
        self._params = params
        self._sim_interface = None  # set by subclass in init_robot()
        self._log = setup_logger(name=name)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize internal vars + robot (synchronous, on this thread)."""
        self.init_vars()
        self.init_robot()
        self.logi("Simulation initialized.")

    def stop(self) -> None:
        """Close the sim interface (idempotent; no-op before `start()`)."""
        if self._sim_interface is None:
            return
        self._sim_interface.close()
        self._sim_interface = None
        self.logi("Stopped.")

    def is_working(self) -> bool:
        """Return True while the sim app is running."""
        return self._sim_interface is not None and self._sim_interface.is_running()

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
    def step(self) -> None:
        """Synchronous sim step — call from **main thread** periodically.

        Processes pending commands → applies them → steps simulation →
        reads state into deques.

        Example:

        ```python
        while robot.is_working():
            robot.set_arm_pos_cmd({"jnt_pos": [...]})
            robot.step()
            state = robot.get_arm_state()
            time.sleep(1.0 / params.ctrl_rate)
        ```
        """
        ...

    # ------------------------------------------------------------------
    # log convenience wrappers
    # ------------------------------------------------------------------

    def loge(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        self._log.error(msg, *args, **kwargs)

    def logd(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        self._log.debug(msg, *args, **kwargs)

    def logi(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        self._log.info(msg, *args, **kwargs)

    def logw(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        self._log.warning(msg, *args, **kwargs)
