"""hex_sim_isaaclab_wrapper — Simulator-backend robot wrapper (Isaac Lab V1).

User-facing API aligned with ``hex_driver_robot``:
  - ``HexRobotSimArcherY6`` / ``HexRobotSimArcherY6Params``

Package layout::

    sim/       — simulator abstraction layer (SimInterface ABC + Isaac Lab impl)
    arm/       — robot arm layer (base classes + Archer Y6)
    utils/     — numpy/torch helpers, dataclass builders
"""

from .arm import HexRobotSimArcherY6, HexRobotSimArcherY6Params

__all__ = [
    "HexRobotSimArcherY6",
    "HexRobotSimArcherY6Params",
]
