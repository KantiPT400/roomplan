"""Synthetic Stray Scanner capture with exact ground truth.

No tape-measured captures were available, so the geometry stage is also tested on rendered captures of a
known flat: two rooms (A 4.20 x 3.35 m, B 2.50 x 3.35 m) joined by a 0.90 m door in a 0.12 m wall, ceiling
2.72 m, a wardrobe (1.20 x 0.60 x 2.00 m) against a wall of A, a table in B. The whole flat is rotated by
17 deg so the Manhattan alignment is exercised.

Depth (256x192, mm, 16-bit PNG) is ray-cast from a camera that walks a loop through both rooms at ~1.45 m,
pitching between -35 and +40 deg (it looks at the ceiling once per room, as the protocol asks), with
  * depth noise: smooth per-frame bias field (0.4% of range, 1 sigma) + pixel jitter 2 mm + 0.2% of range,
    1 mm quantisation, ~0.5% dropped pixels;
  * pose drift: a slow random walk (~1.5 cm and ~0.25 deg per minute on average) added to the written poses.
Intrinsics are those of the sample captures (fx = 1600 px at 1920 x 1440).

usage: python scripts/synth_capture.py <out_dir> [--seed 0] [--frames 1800]
Writes the capture folder and ground_truth.json.
"""
import argparse
import json
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--frames", type=int, default=1800)
ap.add_argument("--no-drift", action="store_true")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

T = 0.12          # wall thickness
H = 2.72          # ceiling
AX, AZ = 4.20, 3.35
BX = 2.50
DOOR = (1.20, 2.10, 2.05)   # door in the wall between A and B: z from 1.20 to 2.10, height 2.05
# boxes: (xmin, ymin, zmin, xmax, ymax, zmax) in the flat's own frame; interior of A: x 0..AX, z 0..AZ;
# B: x AX+T .. AX+T+BX
X0, X1, X2 = 0.0, AX, AX + T + BX
boxes = [
    (-T, -0.1, -T, X2 + T, 0.0, AZ + T),          # floor
    (-T, H, -T, X2 + T, H + 0.1, AZ + T),          # ceiling
    (-T, 0, -T, X2 + T, H, 0.0),                   # wall z=0
    (-T, 0, AZ, X2 + T, H, AZ + T),                # wall z=AZ
    (-T, 0, -T, 0.0, H, AZ + T),                   # wall x=0
    (X2, 0, -T, X2 + T, H, AZ + T),                # wall x=X2
    (X1, 0, 0.0, X1 + T, H, DOOR[0]),              # partition below door
    (X1, 0, DOOR[1], X1 + T, H, AZ),               # partition above door
    (X1, DOOR[2], DOOR[0], X1 + T, H, DOOR[1]),    # lintel
    (0.05, 0, 0.05, 1.25, 2.00, 0.65),             # wardrobe against wall z=0 in A
    (X1 + T + 0.6, 0.72, 1.6, X1 + T + 1.6, 0.76, 2.4),   # table top in B
    (X1 + T + 0.62, 0.0, 1.62, X1 + T + 0.66, 0.72, 1.66),  # table leg
]
B = np.array(boxes)
ROT = 17.0
Rflat = Rot.from_euler("y", ROT, degrees=True).as_matrix()


def raycast(origin, dirs):
    """Nearest hit distance along each unit dir against all AABBs (slab method)."""
    o = origin[None, None, :]
    inv = 1.0 / np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)
    best = np.full(dirs.shape[:2], np.inf)
    for b in B:
        t1 = (b[:3] - o) * inv; t2 = (b[3:] - o) * inv
        tmin = np.max(np.minimum(t1, t2), axis=-1); tmax = np.min(np.maximum(t1, t2), axis=-1)
        hit = (tmax >= np.maximum(tmin, 1e-4))
        t = np.where(tmin > 1e-4, tmin, tmax)
        best = np.where(hit & (t < best), t, best)
    return best


fx_full, cx_full, cy_full = 1600.0, 959.5, 719.5
s = 256 / 1920
fx, cx, cy = fx_full * s, cx_full * s, cy_full * s
W, Hh = 256, 192
u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(Hh) + 0.5)
rays_cam = np.stack([(u - cx) / fx, (v - cy) / fx, np.ones_like(u)], -1)
rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)

# trajectory: loop A -> door -> B -> door -> back to start, in the flat frame
way = np.array([[1.8, 1.5], [3.3, 2.6], [3.6, 1.65], [AX + T + 0.8, 1.65], [AX + T + 1.6, 0.8], [AX + T + 1.8, 2.6],
                [AX + T + 0.8, 1.65], [3.6, 1.65], [2.5, 0.9], [1.0, 2.4], [1.8, 1.5]])
seg = np.linalg.norm(np.diff(way, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]
n = a.frames
dist = np.linspace(0, cum[-1], n)
xz = np.stack([np.interp(dist, cum, way[:, 0]), np.interp(dist, cum, way[:, 1])], 1)
t = np.arange(n) / 46.0
yaw = np.cumsum(np.r_[0, np.full(n - 1, 360 * 2.2 / n)]) + 40 * np.sin(t * 0.9)   # keeps turning, sweeps walls
pitch = -20 + 15 * np.sin(t * 0.5)
for c in (0.18, 0.62):                                     # look up at the ceiling once per room
    k = int(c * n); w = int(0.03 * n)
    pitch[k - w:k + w] = np.maximum(pitch[k - w:k + w], 40 * np.hanning(2 * w))
height = 1.45 + 0.03 * np.sin(t * 2.0)

# drift: random walk in yaw and xz, applied to the written poses only
if a.no_drift:
    dyaw = np.zeros(n); dxz = np.zeros((n, 2))
else:
    dyaw = np.cumsum(rng.normal(0, 0.25 / np.sqrt(60 * 46), n))
    dxz = np.cumsum(rng.normal(0, 0.015 / np.sqrt(60 * 46), (n, 2)), axis=0)

os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)
rows = []
for i in range(n):
    # camera-to-flat rotation: OpenCV camera (x right, y down, z fwd); flat frame y up
    Ryaw = Rot.from_euler("y", yaw[i], degrees=True)
    Rpitch = Rot.from_euler("x", -pitch[i], degrees=True)     # positive pitch looks down
    flip = Rot.from_matrix(np.diag([1.0, -1.0, -1.0]))
    Rc = (Ryaw * flip * Rpitch).as_matrix()                     # cam -> flat
    pos = np.array([xz[i, 0], height[i], xz[i, 1]])
    d = raycast(pos, rays_cam @ Rc.T)
    z = d * rays_cam[..., 2]
    # iPhone LiDAR noise is mostly a smooth per-frame bias field (the 256x192 map is upsampled from a sparse
    # 24x24 dot pattern) plus a few mm of pixel jitter; independent ~2 cm per-pixel noise would turn every
    # wall into a fuzzy 10 cm slab, which real captures do not show.
    field = cv2.resize(rng.normal(0, 1, (6, 8)).astype(np.float32), (W, Hh), interpolation=cv2.INTER_CUBIC)
    z = z * (1 + 0.004 * field) + rng.normal(0, 0.002 + 0.002 * z)
    z[rng.random(z.shape) < 0.005] = 0
    z[~np.isfinite(z) | (z > 6.0)] = 0
    cv2.imwrite(os.path.join(a.out, "depth", f"{i:06d}.png"), np.clip(np.round(z * 1000), 0, 65535).astype(np.uint16))
    # written pose: flat rotated into the "world", plus drift
    Rw = Rflat @ Rc
    pw = Rflat @ pos
    Rd = Rot.from_euler("y", dyaw[i], degrees=True).as_matrix()
    Rw = Rd @ Rw; pw = Rd @ pw + np.array([dxz[i, 0], 0, dxz[i, 1]])
    q = Rot.from_matrix(Rw).as_quat()
    rows.append(f"{t[i]:.6f}, {i:06d}, {pw[0]:.6f}, {pw[1]:.6f}, {pw[2]:.6f}, {q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}, "
                f"{fx_full}, {fx_full}, {cx_full}, {cy_full}, , ")
with open(os.path.join(a.out, "odometry.csv"), "w") as f:
    f.write("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y\n")
    f.write("\n".join(rows) + "\n")
np.savetxt(os.path.join(a.out, "camera_matrix.csv"), np.array([[fx_full, 0, cx_full], [0, fx_full, cy_full], [0, 0, 1]]),
           delimiter=", ", fmt="%.4f")
with open(os.path.join(a.out, "imu.csv"), "w") as f:
    f.write("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n")
gt = {"rooms": {"A": {"dims_m": [AX, AZ], "area_m2": AX * AZ, "ceiling_m": H},
                "B": {"dims_m": [BX, AZ], "area_m2": BX * AZ, "ceiling_m": H}},
      "door": {"width_m": DOOR[1] - DOOR[0], "height_m": DOOR[2]}, "wall_thickness_m": T,
      "flat_rotation_deg": ROT, "seed": a.seed, "frames": n, "drift": not a.no_drift,
      "drift_final": {"yaw_deg": float(dyaw[-1]), "xz_m": dxz[-1].round(4).tolist()},
      "notes": "A has a wardrobe 1.20 x 0.60 x 2.00 m against its z=0 wall (x 0.05-1.25)."}
json.dump(gt, open(os.path.join(a.out, "ground_truth.json"), "w"), indent=1)
print(json.dumps(gt))
