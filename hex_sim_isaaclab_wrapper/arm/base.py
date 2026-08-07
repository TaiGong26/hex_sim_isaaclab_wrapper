"""Base classes for the sim robot wrapper.

## Model

- `HexRobotSimParams` — pure simulation parameters (no hardware concepts).
- `HexRobotSimBase` — abstract base with **work thread** (like
  `hex_driver_robot`'s `HexRobotBase`). Users call `start()` to kick off
  the background thread, then poll state via `get_arm_state()`.
  `stop()` joins the thread and cleans up the simulator.

## Lifecycle

1. `robot = HexRobotSimArcherY6(params)` — constructor.
2. `robot.start()` — calls `init_vars()` + `init_robot()`, then starts
   the background `work_loop()` thread (non-blocking).
3. (background) `work_loop()` runs at `ctrl_rate` via `HexRate`.
4. `robot.stop()` — signals thread to exit, joins, closes sim.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import threading
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
        num_envs:          Number of parallel environments (Isaac Lab).
        device:            Torch device string (e.g. `"cuda:0"`, `"cpu"`).
        headless:          Run without rendering window.
    """
    ctrl_rate: float = 1000.0
    render_rate: float = 60.0
    state_buffer_size: int = 100
    sim_num_envs: int = 1
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
    """Abstract sim robot — work-thread-based lifecycle.

    Example:

    ```python
    robot = HexRobotSimArcherY6(params)
    robot.start()                     # non-blocking, kicks off thread
    while robot.is_working():
        robot.set_arm_pos_cmd({...})
        robot.step()                  # process → sim step → read state
        state = robot.get_arm_state()
        time.sleep(1.0 / params.ctrl_rate)
    robot.stop()                      # joins thread, closes sim
    ```
    """

    def __init__(self, params: HexRobotSimParams, name: str) -> None:
        """Store params, set up logging and the background work thread."""
        self._params = params
        self._sim_interface = None  # set by subclass in init_robot()
        self._log = setup_logger(name=name)

        # Thread lifecycle (mirrors hex_driver_robot's HexRobotBase)
        self._stop_event = threading.Event()
        self._work_thread = threading.Thread(target=self.work_loop, daemon=True)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize internal vars + robot, then start work thread."""
        self.init_vars()
        self.init_robot()
        self._stop_event.clear()
        self._work_thread.start()
        self.logi("Work thread started.")

    def stop(self) -> None:
        """Signal thread to exit, join, and close sim interface."""
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._work_thread.join(timeout=5.0)
        if self._work_thread.is_alive():
            self.logw("Work thread did not exit within 5 s timeout.")
        if self._sim_interface is not None:
            self._sim_interface.close()
            self._sim_interface = None
        self.logi("Stopped.")

    def is_working(self) -> bool:
        """Return True while the sim is running and stop not requested."""
        if not self._sim_interface or not self._sim_interface.is_running():
            return False
        return not self._stop_event.is_set()

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
    def work_loop(self) -> None:
        """Background thread body — called at `ctrl_rate` via `HexRate`.

        > **Note:** Sim stepping should **not** be done here — call `step()`
        from the main thread instead. Isaac Lab's `SimulationContext.step()`
        expects to run on the same thread as the application event loop.
        """
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
