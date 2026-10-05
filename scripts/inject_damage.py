"""Staged-damage test without a staged room: paint a synthetic water stain of KNOWN size onto a measured
wall in every frame the damage detector uses, consistently with the capture's own depth and poses, then
run the pipeline and compare. Checks the detection path (geometry gate, multi-view vote, class, metric
extent) end to end. It does not measure performance on real stains, which we had none of; the report says so.

usage: python scripts/inject_damage.py <lidar_capture> <out_dir> [--w 0.40 --h 0.50 --v 1.10]
"""
import argparse, json, os, subprocess, sys, shutil
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from roomplan.lidar import run
from roomplan.damage import _frames_lidar, _decode, _wall_frame
from roomplan.io import read_depth

ap = argparse.ArgumentParser()
ap.add_argument("capture"); ap.add_argument("out")
ap.add_argument("--w", type=float, default=0.40); ap.add_argument("--h", type=float, default=0.50)
ap.add_argument("--v", type=float, default=1.10, help="stain centre height above floor (m)")
a = ap.parse_args()
plan, dbg = run(a.capture)
scan, R2, fy = dbg["scan"], dbg["R2"], dbg["floor_y"]
frames = _frames_lidar(scan)
# choose the wall whose stain centre is actually seen (depth-consistent) by the most detector frames
def visible_count(w):
    p0, along, nrm = _wall_frame(w, R2)
    c2 = p0 + along * (w.length / 2)
    X = np.array([c2[0], fy + a.v, c2[1]])
    n = 0
    for f in frames:
        R, t = scan.pose(f); Xc = R.T @ (X - t)
        if Xc[2] < 0.4:
            continue
        d = read_depth(scan, f); K = scan.K_depth_for(f, d.shape)
        u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]; v = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
        if 5 <= u < d.shape[1] - 5 and 5 <= v < d.shape[0] - 5 and abs(d[int(v), int(u)] - Xc[2]) < 0.1:
            n += 1
    return n
cands = sorted([w for w in dbg["walls"] if w.length >= 1.0], key=lambda w: -w.length)[:10]
counts = [visible_count(w) for w in cands]
wall = cands[int(np.argmax(counts))]
print("candidate walls seen-by-frames:", counts)
u_c = wall.length / 2
src = os.path.join(a.out, "frames_clean"); dst = os.path.join(a.out, "frames_staged")
os.makedirs(dst, exist_ok=True)
imgs = _decode(os.path.join(a.capture, "rgb.mp4"), frames, src)
p0, along, nrm = _wall_frame(wall, R2)
ax, ay = a.w / 2, a.h / 2
CRACK_V = a.v - ay - 0.25
painted = 0
for f, path in imgs.items():
    img = cv2.imread(path); h, w_ = img.shape[:2]
    d = read_depth(scan, f); dz = cv2.resize(d, (w_, h), interpolation=cv2.INTER_NEAREST)
    K = scan.K_depth_for(f, d.shape).copy(); K[:2] *= w_ / d.shape[1]
    R, t = scan.pose(f)
    v, u = np.mgrid[0:h, 0:w_]
    P = np.stack([(u - K[0, 2]) / K[0, 0] * dz, (v - K[1, 2]) / K[1, 1] * dz, dz], -1) @ R.T + t
    rel = P[..., [0, 2]] - p0
    dist = rel @ nrm; s = rel @ along; hgt = P[..., 1] - fy
    r = np.sqrt(((s - u_c) / ax) ** 2 + ((hgt - a.v) / ay) ** 2)
    alpha = np.clip((1.0 - r) / 0.3, 0, 1) * (np.abs(dist) < 0.04) * (dz > 0.3)   # core r<0.7, fades to r=1
    if alpha.max() > 0:
        painted += 1
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] -= 28 * alpha; lab[..., 2] += 14 * alpha            # darker and yellow-brown
    # second class: a hairline crack, 0.60 m long, 1 cm wide, slightly wavy, below the stain
    cu0 = u_c - 0.30
    wav = 0.01 * np.sin((s - cu0) * 25)
    crack = (s > cu0) & (s < cu0 + 0.60) & (np.abs(hgt - (CRACK_V + wav)) < 0.005) & (np.abs(dist) < 0.04) & (dz > 0.3)
    lab[..., 0] -= 70 * crack
    if alpha.max() == 0 and not crack.any():
        shutil.copy(path, os.path.join(dst, f"{f:06d}.jpg"))       # untouched frames stay byte-identical
    else:
        cv2.imwrite(os.path.join(dst, f"{f:06d}.jpg"),
                    cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
true_area = np.pi * ax * ay * 0.85 ** 2      # ellipse at mid-fade (r = 0.85)
true_crack = {"length_m": 0.60, "u": [round(u_c - 0.30, 2), round(u_c + 0.30, 2)], "v": CRACK_V}
env = dict(os.environ, ROOMPLAN_FRAMES_DIR=dst)
subprocess.run([sys.executable, "-m", "roomplan", a.capture, "--out", os.path.join(a.out, "run")], env=env, check=True,
               cwd=os.path.join(os.path.dirname(__file__), ".."))
name = os.path.basename(os.path.normpath(a.capture))
res = json.load(open(os.path.join(a.out, "run", f"{name}_lidar", "plan.json")))
hits = [d for d in res["damage"]]
summary = {"wall_plane": f"{wall.axis}={wall.coord:.3f}", "frames_painted": painted, "true_area_m2": round(true_area, 4),
           "true_bbox_u": [round(u_c - ax, 2), round(u_c + ax, 2)], "true_bbox_v": [round(a.v - ay, 2), round(a.v + ay, 2)], "true_crack": true_crack,
           "detections": [{k: d[k] for k in ("id", "surface", "class", "extent_m2", "bbox_on_surface_m", "length_m") if k in d} for d in hits]}
json.dump(summary, open(os.path.join(a.out, "inject_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
