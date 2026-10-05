"""LiDAR tier: depth + ARKit poses + intrinsics -> per-room plan."""
from __future__ import annotations
import numpy as np
from .io import load_scan, depth_frame_ids, read_meta
from .fuse import fuse
from .planes import find_floor_ceiling, dominant_angle, rotate_xz
from .walls import extract_walls
from .freespace import carve
from .layout import segment
from .polygon import rectilinear, area
from .uncertainty import length_sigma, ci, Z95, TIER_SYSTEMATIC
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
            selc = inside & (np.abs(Q[:, 1] - cy) < 0.02)
            cl = Q[selc, 1]
            # coverage: share of 10 cm cells under the shrunk footprint that see this ceiling plane.
            # A shelf or loft top is a small horizontal patch; a ceiling spans the room.
            cells = set(map(tuple, np.floor(Q[selc][:, [0, 2]] / 0.1).astype(int).tolist()))
            room_cells = max(Path(Vs).contains_points(_grid_pts(Vs, 0.1)).sum(), 1)
            cov = len(cells) / room_cells
            out["ceiling"] = (float(np.median(cl)), float(1.2533 * cl.std() / np.sqrt(max(len(cl) / 20, 4))),
                              len(cl), float(min(cov, 1.0)))
    return out


def _grid_pts(V, step):
    lo, hi = V.min(0), V.max(0)
    xs, zs = np.meshgrid(np.arange(lo[0], hi[0], step) + step / 2, np.arange(lo[1], hi[1], step) + step / 2)
    return np.stack([xs.ravel(), zs.ravel()], 1)


def run(root, stride=10, voxel=0.015, drift_sigma_per_m=0.0, frames=None, theta=None, drift="on"):
    """LiDAR tier. drift="on" runs the pose-graph correction (drift.py) before mapping; "off" uses the
    ARKit poses as-is (only for the ablation)."""
    scan = load_scan(root)
    meta_in = read_meta(root)
    tier = meta_in.get("tier", "lidar")
    scale_rel = meta_in.get("scale_sigma_rel")
    # pass 1: global frame (floor height, Manhattan angle) from a sparse fuse
    P, F = fuse(scan, stride=stride * 2, voxel=0.03, frames=frames)
    cam = scan.positions()
    cam_y = float(np.median(cam[:, 1]))
    # mono depth smears planes over ~+-10 cm: coarser bins and a lower mass threshold off-LiDAR
    fc = (find_floor_ceiling(P, cam_y, ceil_frac=0.01) if tier == "lidar"
          else find_floor_ceiling(P, cam_y, bin_=0.03, floor_frac=0.02, ceil_frac=0.015))
    if fc["floor"] is None:
        raise RuntimeError("no floor plane found")
    fy = fc["floor"]["y"]
    top = fc["ceiling"]["y"] if fc["ceiling"] else fy + 2.0
    th = dominant_angle(P, fy + 0.4, min(top, fy + 2.0) - 0.2) if theta is None else theta
    drift_info = {"mode": drift}
    if drift == "on":
        from . import drift as D
        segs, breaks, info = D.estimate(scan, fy, min(top, fy + 2.2), th, stride=stride, frames=frames,
                                        tol=0.20 if tier == "lidar" else 0.40,
                                        seg_s=6.0 if tier == "lidar" else 4.0)
        scan.poses = D.corrected_poses(scan, segs, th)
        drift_info.update(info)
        drift_info["max_correction_m"] = float(max(np.hypot(s.dx, s.dz) for s in segs))
        drift_info["yaw_range_deg"] = float(np.ptp([s.yaw for s in segs]))
        # loop-closure edge residual becomes the per-metre drift term of the error model
        drift_sigma_per_m = max(drift_sigma_per_m, 0.5 * (info["rms_edge_residual_x_m"] +
                                                          info["rms_edge_residual_z_m"]) / 5.0)
    P, F = fuse(scan, stride=stride, voxel=voxel, frames=frames)
    cam = scan.positions()
    Q, R2 = rotate_xz(P, th)
    # wall evidence is gathered up to just under the ceiling when one was found (walls seen only high up
    # still count), else up to 2.2 m above the floor
    # mono depth (video/photo) smears a wall over ~+-10 cm: coarser cells and a wider face band there
    wres, wtol = (0.04, 0.06) if tier == "lidar" else (0.08, 0.12)
    walls, ext, lo, res = extract_walls(Q, fy, min(top, fy + 3.2) if fc["ceiling"] else fy + 2.2,
                                        res=wres, face_tol=wtol)
    # which side each wall face was seen from: a room edge may only snap to a face seen from inside it
    from .drift import wall_facing
    posd = dict(zip(scan.poses["frame"].to_numpy(), scan.positions()))
    cam_pts = np.array([posd[f] for f in F])[:, [0, 2]] @ R2.T
    for w, fc_ in zip(walls, wall_facing(walls, Q, cam_pts)):
        w.facing = fc_
    free, occ, g = carve(scan, R2, fy, frames=frames, stride=stride, res=0.03)
    # photos: a few dozen views, so a cell seen free once counts (LiDAR: hundreds of frames, need 2)
    rooms, openings, barrier, space = segment(free, occ, g, walls, grow_min_free=1 if tier == "photo" else None)
    from .layout import refine_jambs
    for o in openings:
        refine_jambs(o, Q, fy)
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
            s = length_sigma(L, se_a, se_b, tier, drift_sigma_per_m * L, sup, scale_rel)
            wall_out.append({"id": f"r{rid}w{i}", "from": [round(a[0], 3), round(a[1], 3)],
                             "to": [round(b[0], 3), round(b[1], 3)], "length_m": ci(L, s),
                             "plane_measured": w is not None, "ends_measured": sup})
        A = area(V)
        per = sum(w["length_m"]["value"] for w in wall_out)
        # area sigma: each edge shifting by its own sigma changes area by ~ sigma * length of that edge
        sA = float(np.sqrt(sum((w["length_m"]["sigma"] * w["length_m"]["value"] / 2) ** 2 for w in wall_out)))
        room = {"id": f"room{rid}", "polygon_m": np.round(V, 3).tolist(), "walls": wall_out,
                "floor_area_m2": ci(A, sA, 2), "perimeter_m": round(per, 2)}
        if "ceiling" in hts and hts["ceiling"][3] >= 0.15:
            cy, cse, cn, cov = hts["ceiling"]
            fy_, fse, fn = hts["floor"]
            Hh = cy - fy_
            # partial coverage: the plane may be a soffit/bulkhead rather than the main ceiling
            cov_term = 0.0 if cov >= 0.4 else 0.03 * (0.4 - cov) / 0.25
            rel = TIER_SYSTEMATIC[tier][1] if scale_rel is None else float(np.hypot(TIER_SYSTEMATIC[tier][1], scale_rel))
            s = float(np.sqrt(cse ** 2 + fse ** 2 + TIER_SYSTEMATIC[tier][0] ** 2 + (rel * Hh) ** 2 + cov_term ** 2))
            room["ceiling_height_m"] = ci(Hh, s)
            room["ceiling_height_m"]["status"] = "measured" if cov >= 0.4 else "inferred"
            room["ceiling_height_m"]["coverage"] = round(cov, 2)
            if Hh < 2.1:
                room["ceiling_height_m"]["note"] = "below 2.1 m: may be a loft or storage platform, verify"
        else:
            room["ceiling_height_m"] = {"value": None, "ci95": [2.3, 3.3], "sigma": None,
                                        "status": "not_observed",
                                        "note": "capture did not see the ceiling over this room; prior range only"}
        out_rooms.append(room)
    ops = []
    for k, o in enumerate(openings):
        if not o.rooms:
            continue
        rel = TIER_SYSTEMATIC[tier][1] if scale_rel is None else float(np.hypot(TIER_SYSTEMATIC[tier][1], scale_rel))
        s = float(np.sqrt(2 * 0.01 ** 2 + o.se ** 2 + TIER_SYSTEMATIC[tier][0] ** 2 + (rel * o.width) ** 2))
        ops.append({"id": f"op{k}", "kind": o.kind, "axis": o.axis, "coord": round(o.coord, 3),
                    "span": [round(o.lo, 3), round(o.hi, 3)], "width_m": ci(o.width, s),
                    "rooms": [f"room{r}" for r in o.rooms]})
    adj = sorted({tuple(op["rooms"]) for op in ops if len(op["rooms"]) == 2})
    meta = {"drift": drift_info, "drift_sigma_per_m": round(drift_sigma_per_m, 5), "tier": tier, "input": meta_in, "frames_used": len(frames) if frames is not None else len(depth_frame_ids(scan, stride)),
            "manhattan_theta_deg": round(float(th), 3), "floor_y": round(fy, 4),
            "global_ceiling": fc["ceiling"], "n_walls": len(walls)}
    sl = Q[(Q[:, 1] > fy + 0.3) & (Q[:, 1] < fy + 1.8)][::4][:, [0, 2]]
    debug = {"plan_points": sl, "scan": scan, "theta": th, "floor_y": fy, "Q": Q, "walls": walls, "rooms_grid": rooms, "grid": g, "R2": R2, "cam_xz": cam[:, [0, 2]] @ R2.T}
    return {"meta": meta, "rooms": out_rooms, "openings": ops,
            "adjacency": [list(a) for a in adj]}, debug
