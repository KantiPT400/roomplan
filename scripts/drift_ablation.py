"""Drift accountability ablation: the same LiDAR capture with the pose-graph correction off and on.

    python scripts/drift_ablation.py <stray_scanner_capture> <out_dir>

Reports (out_dir/drift_ablation.json) and draws (out_dir/drift_ablation.png):
  * loop-closure residual between the first and last 10% of the walk (same-side wall faces matched), in x, z
    and yaw: the capture starts and ends at the same place, so any residual is accumulated drift;
  * wall sharpness (length-weighted median spread of wall faces) and number of wall planes;
  * the two stitched footprints, with wall points coloured by time (early blue, late red).
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from roomplan.io import load_scan, depth_frame_ids
from roomplan.fuse import fuse
from roomplan.planes import find_floor_ceiling, dominant_angle, rotate_xz
from roomplan import drift as D
from roomplan.walls import extract_walls

cap, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
scan = load_scan(cap)
P, F = fuse(scan, stride=20, voxel=0.03)
cam_y = float(np.median(scan.positions()[:, 1]))
fc = find_floor_ceiling(P, cam_y, ceil_frac=0.01)
fy = fc["floor"]["y"]
top = fc["ceiling"]["y"] if fc["ceiling"] else fy + 2.0
top = min(top, fy + 2.2)
th = dominant_angle(P, fy + 0.4, top - 0.2)


def loop_residual(sc):
    ids = depth_frame_ids(sc, 5); n = len(ids)
    A, B = ids[:int(n * 0.10)], ids[-int(n * 0.10):]
    wa, fa = D._seg_walls(sc, A, fy, top, th)
    wb, fb = D._seg_walls(sc, B, fy, top, th)
    dx, wx = D._match_shift(wb, fb, wa, fa, "x", tol=0.25)
    dz, wz = D._match_shift(wb, fb, wa, fa, "z", tol=0.25)
    PA = fuse(sc, frames=A, voxel=0.02)[0]; PB = fuse(sc, frames=B, voxel=0.02)[0]
    ya = dominant_angle(PA, fy + 0.3, top - 0.2); yb = dominant_angle(PB, fy + 0.3, top - 0.2)
    return {"dx_m": round(dx, 4), "dz_m": round(dz, 4), "yaw_deg": round(float(((yb - ya + 45) % 90) - 45), 3),
            "evidence_x": round(wx, 2), "evidence_z": round(wz, 2)}


def stats(sc):
    Pc, Fc = fuse(sc, stride=10, voxel=0.015)
    Q, R2 = rotate_xz(Pc, th)
    walls, *_ = extract_walls(Q, fy, top)
    return Q, Fc, walls, {"wall_planes": len(walls), "wall_sharpness_m": round(D.wall_sharpness(walls), 4),
                          "ghost_walls": D.ghost_walls(walls)}


res = {"capture": os.path.basename(os.path.normpath(cap)), "manhattan_deg": round(float(th), 3)}
res["off"] = {"loop_residual": loop_residual(scan)}
Qoff, Foff, Woff, s = stats(scan); res["off"].update(s)
segs, breaks, info = D.estimate(scan, fy, top, th, stride=10)
scan_on = load_scan(cap)
scan_on.poses = D.corrected_poses(scan_on, segs, th)
res["on"] = {"loop_residual": loop_residual(scan_on), "pose_graph": info,
             "max_correction_m": round(max(float(np.hypot(g.dx, g.dz)) for g in segs), 4),
             "yaw_range_deg": round(float(np.ptp([g.yaw for g in segs])), 3)}
Qon, Fon, Won, s = stats(scan_on); res["on"].update(s)
json.dump(res, open(os.path.join(out, "drift_ablation.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
for ax, (Q, Fr, W, title) in zip(axs, ((Qoff, Foff, Woff, "drift correction OFF (poses as-is)"),
                                      (Qon, Fon, Won, "drift correction ON (pose graph)"))):
    sl = (Q[:, 1] > fy + 0.3) & (Q[:, 1] < fy + 1.8)
    ax.scatter(Q[sl, 0][::3], Q[sl, 2][::3], s=0.15, c=Fr[sl][::3], cmap="coolwarm")
    for w in W:
        if w.axis == "x":
            ax.plot([w.coord] * 2, [w.lo, w.hi], "k-", lw=0.8)
        else:
            ax.plot([w.lo, w.hi], [w.coord] * 2, "k-", lw=0.8)
    lr = res["off" if "OFF" in title else "on"]["loop_residual"]
    ax.set_title(f"{title}\nloop residual dx {lr['dx_m'] * 100:+.1f} cm, dz {lr['dz_m'] * 100:+.1f} cm, "
                 f"yaw {lr['yaw_deg']:+.2f} deg; {len(W)} wall planes")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(alpha=0.2)
fig.suptitle("Wall points coloured by time (blue = start, red = end of the walk)")
fig.tight_layout(); fig.savefig(os.path.join(out, "drift_ablation.png"), dpi=110)
