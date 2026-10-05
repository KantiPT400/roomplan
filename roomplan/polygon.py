"""Rectilinear room polygons snapped to measured wall planes."""
from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage
from .walls import Wall
from .freespace import Grid


def _contour(mask):
    m = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea)[:, 0, :] if cnts else None


def rectilinear(mask, g: Grid, walls: list[Wall], eps_px=2.5, min_edge=0.15, snap=0.25):
    """Return list of edges [(axis, coord, wall_or_None)] around the room, and vertex array (Nx2, x,z)."""
    c = _contour(mask)
    if c is None or len(c) < 4:
        return None, None
    ap = cv2.approxPolyDP(c.reshape(-1, 1, 2), eps_px, True)[:, 0, :]
    pts = g.xz(ap)
    n = len(pts)
    edges = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        d = b - a
        L = np.hypot(*d)
        if L < 1e-6:
            continue
        axis = "z" if abs(d[0]) >= abs(d[1]) else "x"     # mostly-horizontal edge lies at constant z
        coord = (a[1] + b[1]) / 2 if axis == "z" else (a[0] + b[0]) / 2
        edges.append([axis, coord, L])
    # merge consecutive edges with the same orientation (length-weighted coordinate)
    def merge(E):
        out = []
        for e in E:
            if out and out[-1][0] == e[0]:
                p = out[-1]
                p[1] = (p[1] * p[2] + e[1] * e[2]) / (p[2] + e[2]); p[2] += e[2]
            else:
                out.append(list(e))
        if len(out) > 1 and out[0][0] == out[-1][0]:
            p, e = out[-1], out.pop(0)
            p[1] = (p[1] * p[2] + e[1] * e[2]) / (p[2] + e[2]); p[2] += e[2]
        return out
    edges = merge(edges)
    # drop tiny jogs: remove edges shorter than min_edge, then re-merge (keep the longer neighbour's coord)
    for _ in range(4):
        if len(edges) <= 4:
            break
        i = int(np.argmin([e[2] for e in edges]))
        if edges[i][2] >= min_edge:
            break
        prev, nxt = edges[i - 1], edges[(i + 1) % len(edges)]
        keep = prev if prev[2] >= nxt[2] else nxt
        prev[1] = nxt[1] = keep[1]
        edges.pop(i)
        edges = merge(edges)
    if len(edges) < 4 or len(edges) % 2:
        return None, None
    # vertices: intersection of consecutive edges (one x-edge, one z-edge)
    def verts(E):
        V = []
        for i in range(len(E)):
            a, b = E[i - 1], E[i]
            x = a[1] if a[0] == "x" else b[1]
            z = a[1] if a[0] == "z" else b[1]
            V.append((x, z))
        return np.array(V)
    V = verts(edges)
    # snap each edge to the best supporting wall (same axis, close, overlapping span)
    snapped = []
    for i, e in enumerate(edges):
        a, b = V[i], V[(i + 1) % len(V)]
        span = sorted((a[1], b[1])) if e[0] == "x" else sorted((a[0], b[0]))
        best, bs = None, 0
        for w in walls:
            if w.axis != e[0] or abs(w.coord - e[1]) > snap:
                continue
            ov = min(w.hi, span[1]) - max(w.lo, span[0])
            score = ov - 2 * abs(w.coord - e[1])
            if ov > 0.2 and score > bs:
                best, bs = w, score
        if best is not None:
            e[1] = best.coord
        snapped.append(best)
    V = verts(edges)
    return [(e[0], e[1], w) for e, w in zip(edges, snapped)], V


def area(V):
    x, z = V[:, 0], V[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))
