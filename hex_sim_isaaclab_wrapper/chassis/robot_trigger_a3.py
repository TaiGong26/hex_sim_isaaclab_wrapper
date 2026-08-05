"""HexRobotSimTriggerA3 — simulated Trigger A3 chassis (3 omni-wheels)."""

from dataclasses import dataclass

from .base import HexRobotSimChassis, HexRobotSimChassisParams


@dataclass
class HexRobotSimTriggerA3Params(HexRobotSimChassisParams):
    """Parameters for the simulated Trigger A3 chassis."""


class HexRobotSimTriggerA3(HexRobotSimChassis):
    """Simulated Trigger A3 (3 omni-wheels, single drive actuator)."""

    CHASSIS_NAME = "trigger_a3"

    def __init__(
        self,
        params: HexRobotSimTriggerA3Params = HexRobotSimTriggerA3Params(),
    ) -> None:
        super().__init__(params=params, name="Trigger_a3")

    @classmethod
    def _get_articulation_cfg(cls):
        from hex_isaac_usd.configs import HEX_ISAAC_USD_TRIGGER_A3_CFG
        return HEX_ISAAC_USD_TRIGGER_A3_CFG
