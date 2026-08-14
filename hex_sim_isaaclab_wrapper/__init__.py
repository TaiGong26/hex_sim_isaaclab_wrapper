"""hex_sim_isaaclab_wrapper — Simulator-backend robot wrapper (Isaac Lab V1).

User-facing API:
  - `HexRobotSimArcherY6` / `HexRobotSimArcherY6Params`   (arm + grip)
  - `HexRobotSimTriggerA3` / `HexRobotSimTriggerA3Params` (Trigger A3 chassis)
  - `HexRobotSimMaverX4` / `HexRobotSimMaverX4Params`     (Maver X4 chassis)

## Package layout

- `sim/`     — simulator abstraction layer (`SimInterface` ABC + Isaac Lab impls)
- `arm/`     — robot arm layer (base classes + Archer Y6)
- `chassis/` — chassis layer (base class + A3/X4)
- `utils/`   — numpy/torch helpers, dataclass builders
"""

from .arm import HexRobotSimArcherY6, HexRobotSimArcherY6Params
from .chassis import (
    HexRobotSimChassis,
    HexRobotSimChassisParams,
    HexRobotSimMaverX4,
    HexRobotSimMaverX4Params,
    HexRobotSimTriggerA3,
    HexRobotSimTriggerA3Params,
)

__all__ = [
    "HexRobotSimArcherY6",
    "HexRobotSimArcherY6Params",
    "HexRobotSimChassis",
    "HexRobotSimChassisParams",
    "HexRobotSimTriggerA3",
    "HexRobotSimTriggerA3Params",
    "HexRobotSimMaverX4",
    "HexRobotSimMaverX4Params",
]
