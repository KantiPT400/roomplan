"""Back-project depth + poses into one world-frame point cloud."""
from __future__ import annotations
import numpy as np
from .io import Scan, read_depth, depth_frame_ids


def backproject(depth: np.ndarray, K: np.ndarray, min_d=0.3, max_d=4.5, step=3):
    h, w = depth.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    d = depth[0:h:step, 0:w:step]
    m = (d > min_d) & (d < max_d)
    d, u, v = d[m], u[m], v[m]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    return np.stack([x, y, d], -1), np.nonzero(m)


def fuse(scan: Scan, stride=10, min_d=0.3, max_d=4.5, step=3, voxel=0.02, frames=None):
    """Return (points Nx3 world, frame_index N) voxel-downsampled."""
    K = scan.K_depth
    ids = frames if frames is not None else depth_frame_ids(scan, stride)
    pts, fid = [], []
    for f in ids:
        d = read_depth(scan, f)
        P, _ = backproject(d, scan.K_depth_for(f, d.shape), min_d, max_d, step)
        R, t = scan.pose(f)
        pts.append(P @ R.T + t)
        fid.append(np.full(len(P), f))
    P = np.concatenate(pts)
    F = np.concatenate(fid)
    if voxel:
        key = np.floor(P / voxel).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        P, F = P[idx], F[idx]
    return P, F
