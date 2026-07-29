"""hex_sim_isaaclab_wrapper — Simulator-backend robot wrapper (Isaac Lab V1).

User-facing API aligned with ``hex_driver_robot``:
  - ``HexRobotSimArcherY6`` / ``HexRobotSimArcherY6Params``
"""

from .robot_archer_y6 import HexRobotSimArcherY6, HexRobotSimArcherY6Params

__all__ = [
    "HexRobotSimArcherY6",
    "HexRobotSimArcherY6Params",
]