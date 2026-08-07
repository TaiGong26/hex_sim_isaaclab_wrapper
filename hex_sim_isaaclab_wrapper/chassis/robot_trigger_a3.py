"""HexRobotSimTriggerA3 — simulated Trigger A3 chassis (3 omni-wheels).

Canonical joint order (``JOINT_STATE_NAME``) is the three drive wheels only:

    [joint_1, joint_2, joint_3]

The A3 USD carries 48 extra passive ball-caster joints (no actuator). They are
not part of the canonical set and are ignored by the wrapper.
"""

from dataclasses import dataclass

from .base import HexRobotSimChassis, HexRobotSimChassisParams

#: Canonical joint order — the three drive wheels only.
JOINT_STATE_NAME = ["joint_1", "joint_2", "joint_3"]


@dataclass
class HexRobotSimTriggerA3Params(HexRobotSimChassisParams):
    """Parameters for the simulated Trigger A3 chassis."""


class HexRobotSimTriggerA3(HexRobotSimChassis):
    """Simulated Trigger A3 (3 omni-wheels, single drive actuator)."""

    CHASSIS_NAME = "trigger_a3"
    JOINT_STATE_NAME = JOINT_STATE_NAME

    def __init__(
        self,
        params: HexRobotSimTriggerA3Params = HexRobotSimTriggerA3Params(),
    ) -> None:
        super().__init__(params=params, name="Trigger_a3")

    @classmethod
    def _get_articulation_cfg(cls):
        from hex_isaac_usd.configs import HEX_ISAAC_USD_TRIGGER_A3_CFG
        return HEX_ISAAC_USD_TRIGGER_A3_CFG
