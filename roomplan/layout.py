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
        thru = (space & m).sum() / max(m.sum(), 1)
        if thru < 0.5:                       # nothing was seen through it: wall not observed, not an opening
            continue
        sides = []
        for sgn in (-1, 1):
            # probe 0.15-0.45 m off the gap line on each side
            off = np.zeros(2)
            off[0 if o.axis == "x" else 1] = sgn
            labs, freec = [], 0
            for dist in (0.15, 0.3, 0.45):
                for t in np.linspace(o.lo + 0.1, o.hi - 0.1, 5):
                    p = np.array([o.coord, t]) if o.axis == "x" else np.array([t, o.coord])
                    q = g.ij(p + off * dist)
                    if 0 <= q[1] < g.shape[0] and 0 <= q[0] < g.shape[1]:
                        labs.append(int(rooms[q[1], q[0]])); freec += int(space[q[1], q[0]])
            nz = [l for l in labs if l > 0]
            lab = max(set(nz), key=nz.count) if nz else 0
            sides.append((lab, freec / max(len(labs), 1)))
        (la, fa), (lb, fb) = sides
        if la and lb and la != lb:
            o.rooms = (la, lb)
        elif (la or lb) and min(fa, fb) >= 0.5 and la != lb:
            o.rooms = (la or lb,)            # leads to observed space that is not a segmented room
        else:
            continue                          # same room on both sides, or blind on one side: phantom
        o.kind = "door" if o.width <= 1.10 else "passage"
        real.append(o)
    # gaps that were not real openings go back to being open (no barrier) - re-label once
    return rooms, real, barrier, space


def refine_jambs(o: Opening, Q, floor_y, face_tol=0.12, win=0.35, band=(0.3, 1.8)):
    """Re-measure an opening's edges from raw points instead of 4 cm wall cells.

    Wall cells next to a door frame are seen less and fall below the wall test, so cell-based wall ends stop
    short of the frame and openings came out ~10 cm too wide on the synthetic flat (1.00 vs 0.90 m). Here each
    jamb is the extreme along-wall position of points on the wall plane next to the gap (2nd/98th
    percentile), measured at door-handle height. The band (+-12 cm) covers both faces of the wall and the
    reveal: a gap paired from pieces on opposite faces has its coordinate between them.
    """
    ax = 0 if o.axis == "x" else 2      # plane normal axis in Q
    al = 2 if o.axis == "x" else 0      # along-wall axis
    h = Q[:, 1] - floor_y
    on = (np.abs(Q[:, ax] - o.coord) < face_tol) & (h > band[0]) & (h < band[1])
    s = Q[on, al]
    lo_side = s[(s > o.lo - win) & (s < o.lo + 0.10)]
    hi_side = s[(s > o.hi - 0.10) & (s < o.hi + win)]
    new_lo = float(np.percentile(lo_side, 98)) if len(lo_side) > 30 else o.lo
    new_hi = float(np.percentile(hi_side, 2)) if len(hi_side) > 30 else o.hi
    # accept in either direction: cell-based wall ends can stop short of the frame (door too wide) or be
    # bridged into the doorway by the run-closing step (door too narrow)
    if 0.4 < new_hi - new_lo < 1.6 and abs((new_hi - new_lo) - (o.hi - o.lo)) < 0.3:
        o.lo, o.hi = new_lo, new_hi
        o.refined = True
    return o
