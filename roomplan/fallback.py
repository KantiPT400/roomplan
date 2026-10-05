"""Results for any picture: single-view room estimates when photos cannot be registered together.

The photo tier's floor ("any picture in, results out"): a room folder whose photos did not end up in the
mapped component, or a capture that produced no rooms at all, still gets a rectangular room estimate.
Each calibrated photo gives a metric, gravity-aligned point cloud on its own (singleview.py); after Manhattan
alignment its wall points span some extent along the two wall directions. A photo sees only what is in front of
it, so per axis we take the LARGEST extent seen by any photo of that room (axes paired long-with-long). The
estimate is honest about being crude: 30% (1 sigma) on each dimension, status "single_view_estimate", no
adjacency, and it is placed beside the stitched plan, never overlapping it.
"""
from __future__ import annotations
import numpy as np
from .planes import dominant_angle
from .uncertainty import ci

REL_SIGMA = 0.30


def view_extent(v):
    """Sorted (long, short) horizontal extent of the wall points one calibrated view sees, metres."""
    if v.cal is None or v.depth is None:
        return None
    K = v.K.copy(); K[:2] /= 4
    z = v.depth
    h, w = z.shape
    yy, xx = np.mgrid[0:h:2, 0:w:2]
    zz = z[::2, ::2]
    ok = (zz > 0.3) & (zz < 8)
    P = np.stack([(xx - K[0, 2]) / K[0, 0] * zz, (yy - K[1, 2]) / K[1, 1] * zz, zz], -1)[ok]
    G = P @ v.cal["Rg"].T
    floor = -1.40
    band = (G[:, 1] > floor + 0.3) & (G[:, 1] < floor + 2.0)
    if band.sum() < 200:
        return None
    th = dominant_angle(G, floor + 0.3, floor + 2.0)
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    xz = G[band][:, [0, 2]] @ np.array([[c, s], [-s, c]]).T
    # extent including the camera position (0, 0): the room contains the camera
    lo = np.minimum(np.percentile(xz, 2, axis=0), 0); hi = np.maximum(np.percentile(xz, 98, axis=0), 0)
    e = np.sort(hi - lo)[::-1]
    return e


def estimate_rooms(views, groups_missing, x0, z0=0.0, gap=1.0):
    rooms = []
    x = x0
    for g in groups_missing:
        exts = [e for e in (view_extent(v) for v in views if v.group == g) if e is not None]
        if not exts:
            continue
        L = float(max(e[0] for e in exts)); S = float(max(e[1] for e in exts))
        V = np.array([[x, z0], [x + L, z0], [x + L, z0 + S], [x, z0 + S]])
        walls = []
        for i in range(4):
            a, b = V[i], V[(i + 1) % 4]
            Lw = float(np.hypot(*(b - a)))
            walls.append({"id": f"sv_{g}_w{i}", "from": a.round(3).tolist(), "to": b.round(3).tolist(),
                          "length_m": ci(Lw, REL_SIGMA * Lw), "plane_measured": False, "ends_measured": False})
        A = L * S
        rooms.append({"id": f"sv_{g}", "name": g, "polygon_m": V.round(3).tolist(), "walls": walls,
                      "floor_area_m2": ci(A, np.sqrt(2) * REL_SIGMA * A, 2), "perimeter_m": round(2 * (L + S), 2),
                      "ceiling_height_m": {"value": None, "ci95": [2.3, 3.3], "sigma": None, "status": "not_observed"},
                      "status": "single_view_estimate", "stitched": False, "photos_used": len(exts)})
        x += L + gap
    return rooms
