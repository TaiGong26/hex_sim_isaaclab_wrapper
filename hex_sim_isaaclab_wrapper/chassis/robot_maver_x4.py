"""HexRobotSimMaverX4 — simulated Maver X4 chassis (8 MIT motors).

Canonical joint order (``JOINT_STATE_NAME``) is shared with the Mujoco /
URDF / ROS2 references and must not be changed without updating all of them:

    [joint_yaw1, joint_wheel1, joint_yaw2, joint_wheel2,
     joint_yaw3, joint_wheel3, joint_yaw4, joint_wheel4]

- steering (yaw)  → canonical indices [0, 2, 4, 6]
- drive (wheels)  → canonical indices [1, 3, 5, 7]

Sources:
  - ``hex_ros2_dev`` ``hex_ros_sim_maver_x4/mjcf/{robot,setting}.xml``
  - ``hex_ros2_dev`` ``hex_ros_robot_chassis/.../robot_maver.py`` (JOINT_STATE_NAME)
  - ``hex_ros2_dev`` ``hex_ros_urdf_maver_x4/urdf/model.urdf``

The wrapper matches the USD's joints **by name** at spawn time, so the USD
articulation order itself never matters.
"""

from dataclasses import dataclass

from .base import HexRobotSimChassis, HexRobotSimChassisParams

#: Canonical joint order — must match the Mujoco / URDF / ROS2 references.
JOINT_STATE_NAME = [
    "joint_yaw1", "joint_wheel1", "joint_yaw2", "joint_wheel2",
    "joint_yaw3", "joint_wheel3", "joint_yaw4", "joint_wheel4",
]


@dataclass
class HexRobotSimMaverX4Params(HexRobotSimChassisParams):
    """Parameters for the simulated Maver X4 chassis."""


class HexRobotSimMaverX4(HexRobotSimChassis):
    """Simulated Maver X4 (4 steering + 4 drive MIT motors)."""

    CHASSIS_NAME = "maver_x4"
    JOINT_STATE_NAME = JOINT_STATE_NAME

    def __init__(
        self,
        params: HexRobotSimMaverX4Params = HexRobotSimMaverX4Params(),
    ) -> None:
        super().__init__(params=params, name="Maver_x4")

    @classmethod
    def _get_articulation_cfg(cls):
        from hex_isaac_usd.configs import HEX_ISAAC_USD_MAVER_X4_CFG
        return HEX_ISAAC_USD_MAVER_X4_CFG
