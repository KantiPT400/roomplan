"""Manhattan wall extraction from a gravity-aligned point cloud.

A wall is the thing that occupies most of the height between floor and ceiling; furniture and
clutter only occupy a few height bins. We therefore build a 2D map of *vertical extent* (how many
10 cm height bins are occupied per 4 cm cell) and keep high-extent cells as wall evidence.
Axis-aligned runs of wall cells become wall segments, whose plane coordinate is then refined at
1 cm precision from the raw points (median of the face), with a standard error per wall.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2


@dataclass
class Wall:
    axis: str            # "x": wall at constant x (runs along z); "z": wall at constant z (runs along x)
    coord: float         # plane coordinate (m, rotated frame)
    lo: float            # extent along the wall
    hi: float
    n: int = 0           # supporting points
    std: float = 0.0     # spread of the face (m)
    se: float = 0.0      # standard error of coord (m)

    @property
    def length(self):
        return self.hi - self.lo


def extent_map(Q, floor_y, top_y, res=0.04, bin_h=0.10, margin=0.15):
    # robust bounds: a handful of far-away points (bad poses) must not blow up the grid
    lo = Q[:, [0, 2]].min(0) - 0.3
    hi = Q[:, [0, 2]].max(0) + 0.3
    if np.any(hi - lo > 60):                   # not a home: trim stray points from broken poses
        lo = np.percentile(Q[:, [0, 2]], 0.5, axis=0) - 0.5
        hi = np.percentile(Q[:, [0, 2]], 99.5, axis=0) + 0.5
    if np.prod((hi - lo) / res) > 4e7:
        raise RuntimeError(f"site extent {np.round(hi - lo, 1)} m is not a building: poses are broken")
    keep = np.all((Q[:, [0, 2]] >= lo) & (Q[:, [0, 2]] <= hi), axis=1)
    Q = Q[keep]
    shape = np.ceil((hi - lo) / res).astype(int)
    bins = np.arange(floor_y + margin, top_y - margin + 1e-6, bin_h)
    nb = len(bins) - 1
    occ = np.zeros((shape[1], shape[0], nb), bool)
    ij = np.floor((Q[:, [0, 2]] - lo) / res).astype(int)
    hb = np.digitize(Q[:, 1], bins) - 1
    ok = (hb >= 0) & (hb < nb)
    occ[ij[ok, 1], ij[ok, 0], hb[ok]] = True
    return occ.sum(2) / max(nb, 1), lo, res


def _runs(b):
    """Start/end indices of True runs in a 1D bool array."""
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def extract_walls(Q, floor_y, top_y, res=0.04, frac=0.6, min_len=0.35, gap=0.12, face_tol=0.06):
    ext, lo, res = extent_map(Q, floor_y, top_y, res)
    W = (ext >= frac).astype(np.uint8)
    # bridge single-cell holes in the wall evidence
    walls = []
    for axis in ("x", "z"):
        # for x-walls, look at columns (constant x) and find runs along rows (z)
        k = (1, 3) if axis == "x" else (3, 1)  # thicken across the wall by one cell each side
        Wd = cv2.dilate(W, np.ones(k[::-1], np.uint8))
        Wd = cv2.morphologyEx(Wd, cv2.MORPH_CLOSE,
                              np.ones((int(gap / res) + 1, 1) if axis == "x" else (1, int(gap / res) + 1), np.uint8))
        lines = Wd if axis == "x" else Wd.T          # iterate over columns of `lines`
        ncol = lines.shape[1]
        segs = []
        for c in range(ncol):
            for a, b in _runs(lines[:, c] > 0):
                if (b - a) * res >= min_len:
                    segs.append((c, a, b))
        # merge segments in neighbouring columns that overlap along the run
        segs.sort()
        used = [False] * len(segs)
        for i, (c, a, b) in enumerate(segs):
            if used[i]:
                continue
            group = [(c, a, b)]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j, (c2, a2, b2) in enumerate(segs):
                    if used[j]:
                        continue
                    if any(abs(c2 - g[0]) <= 1 and min(b2, g[2]) - max(a2, g[1]) > 0 for g in group):
                        group.append((c2, a2, b2)); used[j] = True; changed = True
            cs = [g[0] for g in group]
            a0, b0 = min(g[1] for g in group), max(g[2] for g in group)
            cmid = (np.mean(cs) + 0.5) * res + (lo[0] if axis == "x" else lo[1])
            alo = a0 * res + (lo[1] if axis == "x" else lo[0])
            ahi = b0 * res + (lo[1] if axis == "x" else lo[0])
            walls.append(Wall(axis, float(cmid), float(alo), float(ahi)))
    # refine each wall coordinate from the raw points of the face
    band = (Q[:, 1] > floor_y + 0.15) & (Q[:, 1] < top_y - 0.15)
    Qb = Q[band]
    out = []
    for w in walls:
        if w.axis == "x":
            sel = (np.abs(Qb[:, 0] - w.coord) < face_tol + 0.04) & (Qb[:, 2] > w.lo) & (Qb[:, 2] < w.hi)
            v = Qb[sel, 0]
        else:
            sel = (np.abs(Qb[:, 2] - w.coord) < face_tol + 0.04) & (Qb[:, 0] > w.lo) & (Qb[:, 0] < w.hi)
            v = Qb[sel, 2]
        if len(v) < 30:
            continue
        # iterate a trimmed median to lock onto the dominant face
        c = np.median(v)
        for _ in range(3):
            vv = v[np.abs(v - c) < face_tol]
            if len(vv) < 30:
                break
            c = np.median(vv)
        w.coord, w.n, w.std = float(c), int(len(vv)), float(vv.std())
        # points are spatially correlated; use an effective n of one per 4 cm of wall length
        neff = max(4, int(w.length / 0.04))
        w.se = float(1.2533 * w.std / np.sqrt(neff))
        out.append(w)
    return out, ext, lo, res
