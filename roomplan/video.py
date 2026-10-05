"""Video tier: a plain handheld clip (no depth, no poses) -> same output contract.

frames (6 fps) -> pycolmap SfM (may fragment) -> mono depth aligned to SfM points -> pseudo-LiDAR
folder (pseudo.py) -> the LiDAR-tier geometry with drift correction, with intervals widened by the
metric-scale prior and the mono-depth shape error.
"""
from __future__ import annotations
import os
import subprocess
from .sfm import run_sfm
from . import pseudo
from .lidar import run as run_geometry

FPS = 6
WIDTH = 960
# iPhone 15 main camera, 26 mm-equivalent: horizontal FOV ~69 deg on a 4:3 frame -> f ~ 0.72 * width.
# Only a prior: SfM refines it.
FOCAL_FRAC = 0.72


def find_video(path):
    if os.path.isfile(path):
        return path
    for f in sorted(os.listdir(path)):
        if f.lower().endswith((".mp4", ".mov", ".m4v")):
            return os.path.join(path, f)
    raise FileNotFoundError("no video in " + path)


def extract_frames(video, out_dir, fps=FPS, width=WIDTH):
    os.makedirs(out_dir, exist_ok=True)
    if any(f.endswith(".jpg") for f in os.listdir(out_dir)):
        return
    subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={fps},scale={width}:-2",
                    "-q:v", "3", os.path.join(out_dir, "f_%05d.jpg")], check=True)


def run(capture, out_dir, fps=FPS):
    work = os.path.join(out_dir, "work")
    img = os.path.join(work, "images")
    extract_frames(find_video(capture), img, fps)
    frags = run_sfm(img, os.path.join(work, "sfm"), focal_px=FOCAL_FRAC * WIDTH, sequential=True, overlap=25)
    pdir = os.path.join(work, "pseudo")
    log = open(os.path.join(work, "pseudo.log"), "w")
    meta = pseudo.build(frags, img, pdir, os.path.join(work, "depth_cache"), "video", fps=fps,
                        log=lambda m: (print(m), log.write(m + "\n")))
    plan, dbg = run_geometry(pdir, stride=1, drift="on")
    plan["meta"]["sfm_fragments"] = len(frags)
    plan["meta"]["frames_registered"] = meta["frames"]
    return plan, dbg
