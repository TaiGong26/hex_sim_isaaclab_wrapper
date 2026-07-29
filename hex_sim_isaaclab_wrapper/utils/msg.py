"""Dataclass builders mirroring ``hex_driver_robot`` helpers."""

from typing import Optional

import numpy as np
from hex_util_msg.dataclass import (
    HexDcBaseHeader,
    HexDcBaseJntFull,
    HexDcBasePose,
    HexDcBaseQuaternion,
    HexDcBaseTime,
    HexDcBaseVector3,
)
from hex_util_runtime import ns_now


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns_to_time(ts_ns: int) -> HexDcBaseTime:
    """Split nanoseconds into secs + nsecs."""
    secs = int(ts_ns // 1_000_000_000)
    nsecs = int(ts_ns % 1_000_000_000)
    return HexDcBaseTime(secs=secs, nsecs=nsecs)


def build_header(ts_ns: Optional[int] = None) -> HexDcBaseHeader:
    """Build a header with the given (or current) nanosecond timestamp."""
    if ts_ns is None:
        ts_ns = ns_now()
    return HexDcBaseHeader(stamp=_ns_to_time(int(ts_ns)))


def build_hex_jnt(
    pos=None,
    vel=None,
    eff=None,
    kp=None,
    kd=None,
    dof: int = 0,
) -> HexDcBaseJntFull:
    """Build a HexDcBaseJntFull from optional arrays, defaulting to zeros."""
    return HexDcBaseJntFull(
        pos=np.asarray(pos) if pos is not None else np.zeros(dof),
        vel=np.asarray(vel) if vel is not None else np.zeros(dof),
        eff=np.asarray(eff) if eff is not None else np.zeros(dof),
        kp=np.asarray(kp) if kp is not None else np.zeros(dof),
        kd=np.asarray(kd) if kd is not None else np.zeros(dof),
    )


def build_pose(
    pos: Optional[np.ndarray] = None,
    quat: Optional[np.ndarray] = None,
) -> HexDcBasePose:
    """Build a HexDcBasePose from optional arrays, defaulting to identity/zero."""
    if pos is None:
        pos = np.zeros(3)
    if quat is None:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    return HexDcBasePose(
        position=HexDcBaseVector3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        orientation=HexDcBaseQuaternion(
            w=float(quat[0]), x=float(quat[1]), y=float(quat[2]), z=float(quat[3]),
        ),
    )


__all__ = [
    "build_header",
    "build_hex_jnt",
    "build_pose",
]
