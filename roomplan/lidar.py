"""LiDAR tier: depth + ARKit poses + intrinsics -> per-room plan."""
from __future__ import annotations
import numpy as np
from .io import load_scan, depth_frame_ids
from .fuse import fuse
from .planes import find_floor_ceiling, dominant_angle, rotate_xz
from .walls import extract_walls
from .freespace import carve
from .layout import segment
from .polygon import rectilinear, area
from .uncertainty import length_sigma, ci, Z95
from matplotlib.path import Path


def room_heights(Q, V, floor_g, shrink=0.25, min_pts=300):
    """Floor and ceiling heights inside polygon V (rotated xz). Ceiling may be unobserved."""
    c = V.mean(0)
    Vs = c + (V - c) * np.clip(1 - shrink / np.maximum(np.abs(V - c).max(0), 1e-3), 0.3, 1)
    inside = Path(Vs).contains_points(Q[:, [0, 2]])
    Y = Q[inside, 1]
    out = {}
    fl = Y[np.abs(Y - floor_g) < 0.06]
    if len(fl) >= min_pts:
        f = np.median(fl)
        fl = fl[np.abs(fl - f) < 0.02]
        out["floor"] = (float(np.median(fl)), float(1.2533 * fl.std() / np.sqrt(max(len(fl) / 20, 4))), len(fl))
    else:
        out["floor"] = (floor_g, 0.01, 0)
    up = Y[(Y > out["floor"][0] + 1.9) & (Y < out["floor"][0] + 3.6)]
    if len(up) >= min_pts:
        h, e = np.histogram(up, np.arange(up.min(), up.max() + 0.02, 0.01))
        hs = np.convolve(h, np.ones(3), "same")
        i = int(np.argmax(hs))
        if hs[i] >= max(min_pts, 0.25 * len(up)):
            y0 = 0.5 * (e[i] + e[i + 1])
            cl = up[np.abs(up - y0) < 0.03]
            cy = np.median(cl)
            cl = cl[np.abs(cl - cy) < 0.02]
            out["ceiling"] = (float(np.median(cl)), float(1.2533 * cl.std() / np.sqrt(max(len(cl) / 20, 4))), len(cl))
    return out


def run(root, stride=10, voxel=0.015, drift_sigma_per_m=0.0, frames=None, theta=None):
    scan = load_scan(root)
    P, F = fuse(scan, stride=stride, voxel=voxel, frames=frames)
    cam = scan.positions()
    cam_y = float(np.median(cam[:, 1]))
    fc = find_floor_ceiling(P, cam_y)
    if fc["floor"] is None:
        raise RuntimeError("no floor plane found")
    fy = fc["floor"]["y"]
    top = fc["ceiling"]["y"] if fc["ceiling"] else fy + 2.0
    th = dominant_angle(P, fy + 0.4, min(top, fy + 2.0) - 0.2) if theta is None else theta
    Q, R2 = rotate_xz(P, th)
    walls, ext, lo, res = extract_walls(Q, fy, min(top, fy + 2.2))
    free, occ, g = carve(scan, R2, fy, frames=frames, stride=stride, res=0.03)
    rooms, openings, barrier, space = segment(free, occ, g, walls)
    out_rooms = []
    for rid in range(1, rooms.max() + 1):
        edges, V = rectilinear(rooms == rid, g, walls)
        if edges is None:
            continue
        hts = room_heights(Q, V, fy)
        wall_out = []
        n = len(V)
        for i, (axis, coord, w) in enumerate(edges):
            a, b = V[i], V[(i + 1) % n]
            L = float(np.hypot(*(b - a)))
            # the two perpendicular edges that terminate this one
            pa, pb = edges[i - 1][2], edges[(i + 1) % n][2]
            se_a = pa.se if pa else 0.0
            se_b = pb.se if pb else 0.0
            sup = (pa is not None) and (pb is not None)
            s = length_sigma(L, se_a, se_b, "lidar", drift_sigma_per_m * L, sup)
            wall_out.append({"id": f"r{rid}w{i}", "from": [round(a[0], 3), round(a[1], 3)],
                             "to": [round(b[0], 3), round(b[1], 3)], "length_m": ci(L, s),
                             "plane_measured": w is not None, "ends_measured": sup})
        A = area(V)
        per = sum(w["length_m"]["value"] for w in wall_out)
        # area sigma: each edge shifting by its own sigma changes area by ~ sigma * length of that edge
        sA = float(np.sqrt(sum((w["length_m"]["sigma"] * w["length_m"]["value"] / 2) ** 2 for w in wall_out)))
        room = {"id": f"room{rid}", "polygon_m": np.round(V, 3).tolist(), "walls": wall_out,
                "floor_area_m2": ci(A, sA, 2), "perimeter_m": round(per, 2)}
        if "ceiling" in hts:
            cy, cse, cn = hts["ceiling"]
            fy_, fse, fn = hts["floor"]
            Hh = cy - fy_
            s = float(np.sqrt(cse ** 2 + fse ** 2 + 0.005 ** 2 + (0.004 * Hh) ** 2))
            room["ceiling_height_m"] = ci(Hh, s)
            room["ceiling_height_m"]["status"] = "measured"
        else:
            room["ceiling_height_m"] = {"value": None, "ci95": [2.3, 3.3], "sigma": None,
                                        "status": "not_observed",
                                        "note": "capture did not see the ceiling over this room; prior range only"}
        out_rooms.append(room)
    ops = []
    for k, o in enumerate(openings):
        if not o.rooms:
            continue
        s = float(np.sqrt(2 * 0.01 ** 2 + o.se ** 2 + 0.005 ** 2))
        ops.append({"id": f"op{k}", "kind": o.kind, "axis": o.axis, "coord": round(o.coord, 3),
                    "span": [round(o.lo, 3), round(o.hi, 3)], "width_m": ci(o.width, s),
                    "rooms": [f"room{r}" for r in o.rooms]})
    adj = sorted({tuple(op["rooms"]) for op in ops if len(op["rooms"]) == 2})
    meta = {"tier": "lidar", "frames_used": len(frames) if frames is not None else len(depth_frame_ids(scan, stride)),
            "manhattan_theta_deg": round(float(th), 3), "floor_y": round(fy, 4),
            "global_ceiling": fc["ceiling"], "n_walls": len(walls)}
    debug = {"Q": Q, "walls": walls, "rooms_grid": rooms, "grid": g, "R2": R2, "cam_xz": cam[:, [0, 2]] @ R2.T}
    return {"meta": meta, "rooms": out_rooms, "openings": ops,
            "adjacency": [list(a) for a in adj]}, debug
