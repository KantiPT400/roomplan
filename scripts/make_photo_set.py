"""Build a photo-tier test set (one folder per room) from a LiDAR capture's video.

We had no separately shot photo sets, so this simulates what the capture protocol asks a person to do,
using the LiDAR-tier plan only to know which room each video frame was taken in:
  * per room: up to N stills taken from inside the room, spread across viewing directions, pitched
    between -40 and +15 degrees (a person photographing walls, not the floor), sharpest candidates only;
  * per opening joining two rooms: the frame closest to the doorway goes into both rooms' folders
    (protocol step 4: the doorway shot that ties rooms together).
Frames come from the capture's own rgb.mp4 at full resolution. The test therefore shares the device and
lighting of the LiDAR reference, which flatters the photo tier somewhat; the report says so.

usage: python scripts/make_photo_set.py <lidar_capture_dir> <lidar plan.json> <out_dir> [--per-room 6]
"""
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("capture"); ap.add_argument("plan"); ap.add_argument("out")
ap.add_argument("--per-room", type=int, default=6)
ap.add_argument("--manifest", help="re-create an existing set exactly from its manifest.json (frame numbers)")
a = ap.parse_args()

plan = json.load(open(a.plan)) if not a.manifest else {"meta": {"manhattan_theta_deg": 0.0}, "rooms": [], "openings": []}
th = plan["meta"]["manhattan_theta_deg"]
c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
R2 = np.array([[c, s], [-s, c]])
od = pd.read_csv(os.path.join(a.capture, "odometry.csv")); od.columns = [x.strip() for x in od.columns]
pos = od[[ "x", "z"]].to_numpy() @ R2.T
q = Rotation.from_quat(od[["qx", "qy", "qz", "qw"]].to_numpy())
fwd = q.apply([0, 0, 1.0])
pitch = np.degrees(np.arcsin(-fwd[:, 1]))          # + looking down
yaw = np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2]))
ok_pitch = (pitch < 40) & (pitch > -15)
# sharpness proxy: low angular + linear speed
t = od["timestamp"].to_numpy()
spd = np.r_[0, np.linalg.norm(np.diff(od[["x", "y", "z"]].to_numpy(), axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)]
rot = np.r_[0, np.degrees((q[1:] * q[:-1].inv()).magnitude()) / np.maximum(np.diff(t), 1e-3)]
calm = (spd < np.percentile(spd, 60)) & (rot < np.percentile(rot, 50))

sel = {}
# Protocol (docs/capture_protocol.md, photo tier): walk slowly around each room taking a photo every
# half step or ~30 deg of turn, phone upright at chest height, some floor in every shot, so that each
# photo overlaps the previous one by about half; 2-8 photos per room. Simulated here by sampling the
# walkthrough inside each room at equal "view change" (rotation/30 deg + translation/0.6 m), which keeps
# consecutive photos overlapping like the protocol asks.
for r in plan["rooms"]:
    V = np.array(r["polygon_m"]); cen = V.mean(0)
    inside = Path(cen + (V - cen) * 0.95).contains_points(pos)
    idx = np.where(inside & ok_pitch)[0]
    if len(idx) < 10:
        continue
    # longest time-contiguous run inside the room (gaps < 1 s allowed)
    runs, cur = [], [idx[0]]
    for k in idx[1:]:
        if t[k] - t[cur[-1]] < 1.0:
            cur.append(k)
        else:
            runs.append(cur); cur = [k]
    runs.append(cur)
    run = max(runs, key=len)
    run = [k for k in run if calm[k]] or run
    step = [0.0]
    for a_, b_ in zip(run[:-1], run[1:]):
        dr = np.degrees((q[b_] * q[a_].inv()).magnitude())
        step.append(dr / 30.0 + np.linalg.norm(od.loc[b_, ["x", "y", "z"]].to_numpy(float) -
                                               od.loc[a_, ["x", "y", "z"]].to_numpy(float)) / 0.6)
    cum = np.cumsum(step)
    m = int(min(8, max(2, np.floor(cum[-1]) + 1)))
    targets = np.linspace(0, cum[-1], m)
    picks = [int(run[int(np.argmin(np.abs(cum - tt)))]) for tt in targets]
    sel[r.get("name", r["id"]).replace(" ", "_")] = sorted(set(picks))
for o in plan["openings"]:
    if len(o["rooms"]) != 2:
        continue
    p = np.array([o["coord"], np.mean(o["span"])]) if o["axis"] == "x" else np.array([np.mean(o["span"]), o["coord"]])
    d = np.linalg.norm(pos - p, axis=1)
    cand = np.where((d < 1.2) & ok_pitch)[0]
    if len(cand):
        i = cand[np.argmin(d[cand])]
        names = {r["id"]: r.get("name", r["id"]).replace(" ", "_") for r in plan["rooms"]}
        # one doorway shot per opening, filed under one room only (as a person would)
        target = names[o["rooms"][1]]
        sel.setdefault(target, []).append(i)
video = os.path.join(a.capture, "rgb.mp4")
manifest = {room: sorted(set(int(f) for f in frames)) for room, frames in sel.items() if len(frames) >= 2}
if a.manifest:
    manifest = json.load(open(a.manifest))["frames"]
need = sorted(set(sum(manifest.values(), [])))
tmp = os.path.join(a.out, "_frames"); os.makedirs(tmp, exist_ok=True)
# one decoding pass: select every needed frame, written in order, then renamed by frame number
expr = "+".join("eq(n\\,%d)" % i for i in need)
subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", "select=" + expr, "-vsync", "0", "-q:v", "2",
                os.path.join(tmp, "o_%05d.jpg")], check=True)
import shutil
for k, i in enumerate(need):
    os.replace(os.path.join(tmp, "o_%05d.jpg" % (k + 1)), os.path.join(tmp, "%05d.jpg" % i))
import cv2
# A phone stores photos upright (EXIF orientation). The capture app stored sensor-oriented frames, so
# rotate each by the multiple of 90 deg that puts world-up at the top of the image.
up_cam = q.inv().apply(np.tile([0, 1.0, 0], (len(q), 1)))      # world up in camera coords (x right, y down)
for room, frames in manifest.items():
    d = os.path.join(a.out, room); os.makedirs(d, exist_ok=True)
    for i in frames:
        img = cv2.imread(os.path.join(tmp, "%05d.jpg" % i))
        ux, uy = up_cam[i][0], up_cam[i][1]
        if abs(uy) >= abs(ux):
            img = img if uy < 0 else cv2.rotate(img, cv2.ROTATE_180)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if ux < 0 else cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(d, "IMG_%05d.jpg" % i), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
shutil.rmtree(tmp)
json.dump({"source": a.capture, "plan": a.plan, "frames": manifest}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
print({k: len(v) for k, v in manifest.items()})
