"""Utility functions shared across the wrapper."""

from typing import Optional

import numpy as np
from hex_util_msg.dataclass import (
    HexDcBaseHeader,
    HexDcBaseJntFull,
    HexDcBaseJntState,
    HexDcBasePose,
    HexDcBaseQuaternion,
    HexDcBaseTime,
    HexDcBaseVector3,
)
from hex_util_runtime import ns_now


# ---------------------------------------------------------------------------
# Numpy ↔ torch
# ---------------------------------------------------------------------------

def torch_to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray (detached, CPU)."""
    return tensor.detach().cpu().numpy()


def numpy_to_torch(arr: np.ndarray, device: str = "cpu"):
    """Convert a numpy ndarray to a torch tensor on the given device."""
    import torch
    return torch.from_numpy(arr).to(device)


# ---------------------------------------------------------------------------
# Quaternion helper  (wxyz format throughout)
# ---------------------------------------------------------------------------

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by quaternion *q* (wxyz format).

    Equivalent to ``p * v * p^{-1}`` where *p* is the unit quaternion.
    """
    # w, x, y, z
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]

    uv_x = 2.0 * (y * vz - z * vy)
    uv_y = 2.0 * (z * vx - x * vz)
    uv_z = 2.0 * (x * vy - y * vx)

    uuv_x = 2.0 * (w * uv_x + y * uv_z - z * uv_y)
    uuv_y = 2.0 * (w * uv_y + z * uv_x - x * uv_z)
    uuv_z = 2.0 * (w * uv_z + x * uv_y - y * uv_x)

    return np.array([vx + uv_x + uuv_x, vy + uv_y + uuv_y, vz + uv_z + uuv_z])


# ---------------------------------------------------------------------------
# Dataclass builders  (mirror hex_driver_robot helpers)
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