"""2D free-space carving.

Rays from the camera to every wall-height depth hit mark cells as free; the hit cell is occupied.
Votes are counted per frame (a cell seen free by 12 frames has free=12), which makes the result
robust to a few bad depth pixels and independent of how long the camera lingers in one place.
The room interior is then the free region connected to the camera path, bounded by occupied cells.
"""
from __future__ import annotations
import numpy as np
import cv2
from .io import Scan, read_depth, depth_frame_ids
from .fuse import backproject


class Grid:
    def __init__(self, lo, hi, res):
        self.lo, self.res = np.asarray(lo, float), float(res)
        self.shape = tuple(np.ceil((np.asarray(hi) - self.lo) / self.res).astype(int)[::-1])  # rows=z, cols=x

    def ij(self, xz):
        return np.floor((np.asarray(xz) - self.lo) / self.res).astype(int)  # (col,row)

    def xz(self, ij):
        return (np.asarray(ij) + 0.5) * self.res + self.lo


def wall_hits(scan: Scan, frame: int, floor_y: float, band=(-0.05, 3.5), max_d=4.5, step=2):
    d = read_depth(scan, frame)
    P, _ = backproject(d, scan.K_depth_for(frame, d.shape), 0.3, max_d, step)
    R, t = scan.pose(frame)
    W = P @ R.T + t
    sel = (W[:, 1] > floor_y + band[0]) & (W[:, 1] < floor_y + band[1])
    return W[sel][:, [0, 2]], t[[0, 2]]


def carve(scan: Scan, R2: np.ndarray, floor_y: float, frames=None, stride=10, res=0.03,
          band=(-0.05, 3.5), max_d=4.5, pad=0.5):
    ids = frames if frames is not None else depth_frame_ids(scan, stride)
    per = []
    for f in ids:
        H, c = wall_hits(scan, f, floor_y, band, max_d)
        if len(H):
            per.append((H @ R2.T, c @ R2.T))
    allp = np.concatenate([h for h, _ in per] + [c[None] for _, c in per])
    g = Grid(allp.min(0) - pad, allp.max(0) + pad, res)
    free = np.zeros(g.shape, np.int32)
    occ = np.zeros(g.shape, np.int32)
    for H, c in per:
        hi = g.ij(H)
        ci = tuple(g.ij(c))
        ray = np.zeros(g.shape, np.uint8)
        hit = np.zeros(g.shape, np.uint8)
        for hx, hz in np.unique(hi, axis=0):
            cv2.line(ray, ci, (int(hx), int(hz)), 1, 1)
            hit[hz, hx] = 1
        free += ray
        occ += hit
    return free, occ, g
