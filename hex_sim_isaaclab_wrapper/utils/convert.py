"""Torch/numpy conversion helpers and quaternion math."""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Numpy ↔ torch
# ---------------------------------------------------------------------------

def torch_to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray (detached, CPU)."""
    return tensor.detach().cpu().numpy()


def numpy_to_torch(arr: np.ndarray, device: str = "cpu"):
    """Convert a numpy ndarray to a torch tensor on the given device."""
    return torch.from_numpy(arr.astype(np.float32)).to(device)



# ---------------------------------------------------------------------------
# Quaternion helper  (wxyz format throughout)
# ---------------------------------------------------------------------------

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by quaternion *q* (wxyz format).

    Equivalent to ``p * v * p^{-1}`` where *p* is the unit quaternion.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]

    uv_x = 2.0 * (y * vz - z * vy)
    uv_y = 2.0 * (z * vx - x * vz)
    uv_z = 2.0 * (x * vy - y * vx)

    uuv_x = 2.0 * (w * uv_x + y * uv_z - z * uv_y)
    uuv_y = 2.0 * (w * uv_y + z * uv_x - x * uv_z)
    uuv_z = 2.0 * (w * uv_z + x * uv_y - y * uv_x)

    return np.array([vx + uv_x + uuv_x, vy + uv_y + uuv_y, vz + uv_z + uuv_z])


__all__ = [
    "torch_to_numpy",
    "numpy_to_torch",
    "quat_rotate",
]
