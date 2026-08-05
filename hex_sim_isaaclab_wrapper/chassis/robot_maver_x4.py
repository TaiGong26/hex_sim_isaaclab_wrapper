"""HexRobotSimMaverX4 — simulated Maver X4 chassis (8 MIT motors)."""

from dataclasses import dataclass

from .base import HexRobotSimChassis, HexRobotSimChassisParams


@dataclass
class HexRobotSimMaverX4Params(HexRobotSimChassisParams):
    """Parameters for the simulated Maver X4 chassis."""


class HexRobotSimMaverX4(HexRobotSimChassis):
    """Simulated Maver X4 (4 steering + 4 drive MIT motors)."""

    CHASSIS_NAME = "maver_x4"

    def __init__(
        self,
        params: HexRobotSimMaverX4Params = HexRobotSimMaverX4Params(),
    ) -> None:
        super().__init__(params=params, name="Maver_x4")

    @classmethod
    def _get_articulation_cfg(cls):
        from hex_isaac_usd.configs import HEX_ISAAC_USD_MAVER_X4_CFG
        return HEX_ISAAC_USD_MAVER_X4_CFG
