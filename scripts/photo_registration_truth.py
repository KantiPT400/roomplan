"""Score every pairwise photo registration against the true relative pose.

The photo test sets are cut from a LiDAR capture's own video (scripts/make_photo_set.py), so every photo
has an ARKit pose: a free ground truth for the photo tier's registration step.

    python scripts/photo_registration_truth.py <photo_set_dir> <lidar_capture_dir> <work_dir> <out.csv>
"""
import os, sys
import numpy as np, cv2, pandas as pd
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import roomplan.photo as ph
import roomplan.mvreg as mv

photos, capture, work, out_csv = sys.argv[1:5]
od = pd.read_csv(os.path.join(capture, "odometry.csv")); od.columns = [c.strip() for c in od.columns]
items = ph.collect(photos, os.path.join(work, "images"))
os.makedirs(os.path.join(work, "disp_cache"), exist_ok=True)
views = [mv.View(it["name"], cv2.imread(os.path.join(work, "images", it["name"])), ph._K(it), it["room"],
                 os.path.join(work, "disp_cache")) for it in items]
fr = [int(it["name"].split("IMG_")[1][:5]) for it in items]
Q = Rot.from_quat(od[["qx", "qy", "qz", "qw"]].to_numpy(float))
P = od[["x", "y", "z"]].to_numpy(float)
F = np.diag([1, -1, -1.0])            # ARKit camera (x right, y up, z back) -> OpenCV camera
rows = []
for a in range(len(views)):
    for b in range(len(views)):
        if a == b:
            continue
        r = mv.register(views[a], views[b])
        if r is None:
            continue
        Ra = Q[fr[a]].as_matrix() @ F; Rb = Q[fr[b]].as_matrix() @ F
        err = np.degrees(Rot.from_matrix(r["R"] @ (Rb.T @ Ra).T).magnitude())
        rows.append({"a": items[a]["name"], "b": items[b]["name"], "room_a": items[a]["room"], "room_b": items[b]["room"],
                     "method": r["method"], "inliers": r["inliers"], "rot_err_deg": round(float(err), 2),
                     "floor_frac_inliers": round(float(np.mean(r["pa"][:, 1] > 0.6 * views[a].h)), 3),
                     "true_dist_m": round(float(np.linalg.norm(P[fr[a]] - P[fr[b]])), 3)})
df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
good = (df.rot_err_deg < 5).sum(); bad = (df.rot_err_deg > 15).sum()
print(f"registrations {len(df)}, within 5 deg {good} ({good / max(len(df), 1):.0%}), off by >15 deg {bad}, "
      f"median error {df.rot_err_deg.median():.1f} deg")
