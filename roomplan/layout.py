"""Rooms, openings and adjacency from wall segments + carved free space.

Rooms are regions of observed free space separated by walls. Wall segments are drawn as barriers;
doorway-sized gaps between walls are closed with "opening" barriers that remember which two
regions they join. Each closed gap therefore yields both an opening and an adjacency edge.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2
from scipy import ndimage
from .walls import Wall
from .freespace import Grid


@dataclass
class Opening:
    axis: str           # axis of the wall it sits in
    coord: float        # wall plane coordinate
    lo: float           # extent along the wall
    hi: float
    kind: str = "door"  # door | passage | window
    rooms: tuple = ()
    se: float = 0.0

    @property
    def width(self):
        return self.hi - self.lo


def _seg_px(g: Grid, axis, coord, lo, hi):
    if axis == "x":
        a, b = g.ij([coord, lo]), g.ij([coord, hi])
    else:
        a, b = g.ij([lo, coord]), g.ij([hi, coord])
    return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))


def find_gaps(walls: list[Wall], min_w=0.55, max_w=1.40, tol=0.15, min_jamb=0.5):
    """Doorway-sized gaps: between collinear walls, or between a wall end and a perpendicular wall.

    Walls shorter than `min_jamb` cannot form a doorway: short stubs are usually open door leaves,
    shelving ends or furniture, and treating them as jambs creates phantom openings."""
    walls = [w for w in walls if w.length >= min_jamb]
    gaps = []
    for i, a in enumerate(walls):
        for b in walls[i + 1:]:
            if a.axis == b.axis and abs(a.coord - b.coord) < tol:
                lo, hi = (a.hi, b.lo) if a.hi < b.lo else (b.hi, a.lo)
                if min_w <= hi - lo <= max_w:
                    c = (a.coord * a.n + b.coord * b.n) / max(a.n + b.n, 1)
                    gaps.append(Opening(a.axis, c, lo, hi, se=float(np.hypot(a.se, b.se))))
    for a in walls:
        for b in walls:
            if a.axis == b.axis:
                continue
            # does b's line cross a's extension, and is a's coord inside b's span?
            if not (b.lo - tol <= a.coord <= b.hi + tol):
                continue
            for end, sgn in ((a.hi, 1), (a.lo, -1)):
                d = (b.coord - end) * sgn
                if min_w <= d <= max_w:
                    lo, hi = sorted((end, b.coord))
                    gaps.append(Opening(a.axis, a.coord, lo, hi, se=float(np.hypot(a.se, b.se))))
    # de-duplicate overlapping gaps on the same line, keep the narrowest
    gaps.sort(key=lambda o: o.width)
    keep = []
    for o in gaps:
        if not any(k.axis == o.axis and abs(k.coord - o.coord) < tol and
                   min(k.hi, o.hi) - max(k.lo, o.lo) > 0.2 for k in keep):
            keep.append(o)
    return keep


def segment(free, occ, g: Grid, walls: list[Wall], min_free=2, min_area=1.2, wall_px=2):
    barrier = np.zeros(g.shape, np.uint8)
    for w in walls:
        p, q = _seg_px(g, w.axis, w.coord, w.lo, w.hi)
        cv2.line(barrier, p, q, 1, wall_px)
    gaps = find_gaps(walls)
    gap_mask = np.zeros(g.shape, np.uint8)
    for k, o in enumerate(gaps):
        p, q = _seg_px(g, o.axis, o.coord, o.lo, o.hi)
        cv2.line(gap_mask, p, q, k + 1, wall_px)
    space = (free >= min_free)
    space = cv2.morphologyEx(space.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    open_ = space & (barrier == 0) & (gap_mask == 0)
    lab, n = ndimage.label(open_)
    areas = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * g.res ** 2
    rooms = np.zeros_like(lab)
    k = 0
    for i, a in enumerate(areas, 1):
        if a >= min_area:
            k += 1
            rooms[lab == i] = k
    # openings: which rooms touch each side of each gap
    real = []
    for idx, o in enumerate(gaps):
        m = gap_mask == idx + 1
        ring = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        touching = sorted(set(np.unique(rooms[ring & (rooms > 0)]).tolist()))
        # an opening must have observed free space through it
        thru = (space & m).sum() / max(m.sum(), 1)
        if thru < 0.5:
            continue
        o.rooms = tuple(touching)
        o.kind = "door" if o.width <= 1.10 else "passage"
        real.append(o)
    # gaps that were not real openings go back to being open (no barrier) - re-label once
    return rooms, real, barrier, space
