"""Video tier: a plain handheld clip (no depth, no poses) -> same output contract.

First version used pycolmap incremental SfM; on the sample walkthroughs it fragmented into 7-17 models
(white walls, motion blur) and the fragments could not be chained reliably. This version reuses the photo
tier's machinery, which does not need long feature tracks:

  frames (3 fps, upright) -> per-frame metric depth + gravity (singleview.py) -> each frame registered to
  the next few frames (essential matrix, floor features masked, translation from metric depth) ->
  global heading/position solve (mvreg.refine) -> pseudo-LiDAR folder -> shared geometry with the
  drift pose graph (fragment breaks respected) -> intervals widened by the camera-height scale prior.

Orientation: phone videos normally carry a rotation tag that ffmpeg applies; capture-app exports may be
sensor-oriented (the sample rgb.mp4 is). The upright rotation is chosen once per clip as the one under which
single-view floor detection succeeds most often on a few probe frames.
"""
from __future__ import annotations
import os
import subprocess
import numpy as np
import cv2
from .mvreg import View, layout, write_pseudo
from .singleview import calibrate
from .mono import disparity
from .lidar import run as run_geometry

FPS = 3
WIDTH = 960
FOCAL_FRAC = 0.80        # iPhone 1x video, focal / long side (prior; sample capture: 0.833)
NEIGHBOURS = 4


def find_video(path):
    if os.path.isfile(path):
        return path
    for f in sorted(os.listdir(path)):
        if f.lower().endswith((".mp4", ".mov", ".m4v")):
            return os.path.join(path, f)
    raise FileNotFoundError("no video in " + path)


def extract_frames(video, out_dir, fps=FPS, width=WIDTH):
    os.makedirs(out_dir, exist_ok=True)
    if not any(f.endswith(".jpg") for f in os.listdir(out_dir)):
        subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={fps},scale='if(gt(iw,ih),{width},-2)':'if(gt(iw,ih),-2,{width})'",
                        "-q:v", "3", os.path.join(out_dir, "f_%05d.jpg")], check=True)
    return sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))


ROTS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _K(w, h):
    f = FOCAL_FRAC * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def choose_rotation(img_dir, names, n_probe=8):
    """Upright rotation for the clip: the one in which the detected floor sits lowest in the image.

    'Floor found at all' is not decisive: a walkthrough pitched down 30 deg sees floor in every rotation.
    In an upright image floor pixels concentrate at the bottom (mean row ~0.7-0.8 of the height); upside
    down at the top; sideways around the middle.
    """
    probe = names[:: max(1, len(names) // n_probe)][:n_probe]
    score = {}
    for deg, code in ROTS.items():
        rows = []
        for nm in probe:
            img = cv2.imread(os.path.join(img_dir, nm))
            if code is not None:
                img = cv2.rotate(img, code)
            h, w = img.shape[:2]
            d = cv2.resize(disparity(img), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            K = _K(w, h); K[:2] /= 4
            c = calibrate(d, K)
            if c is None:
                continue
            from .singleview import depth_from
            z = depth_from(d, c)
            hh, ww = z.shape
            vv, uu = np.mgrid[0:hh, 0:ww]
            P = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], -1)
            yg = (P @ c["Rg"].T)[..., 1]
            fl = yg < yg.min() + 0.15 * (np.percentile(yg, 95) - yg.min()) + 0.1
            if fl.any():
                rows.append(vv[fl].mean() / hh)
        score[deg] = float(np.mean(rows)) if rows else 0.0
    return max(score, key=score.get), {k: round(v, 2) for k, v in score.items()}


def run(capture, out_dir, fps=FPS, log=print):
    work = os.path.join(out_dir, "work")
    raw = os.path.join(work, "frames")
    names = extract_frames(find_video(capture), raw, fps)
    rot, score = choose_rotation(raw, names)
    log(f"frames {len(names)}, upright rotation {rot} deg (probe floor hits {score})")
    up = os.path.join(work, "images")
    os.makedirs(up, exist_ok=True)
    cache = os.path.join(work, "disp_cache")
    os.makedirs(cache, exist_ok=True)
    views = []
    for nm in names:
        dst = os.path.join(up, nm)
        if not os.path.exists(dst):
            img = cv2.imread(os.path.join(raw, nm))
            if ROTS[rot] is not None:
                img = cv2.rotate(img, ROTS[rot])
            cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img = cv2.imread(dst)
        h, w = img.shape[:2]
        views.append(View(nm, img, _K(w, h), "video", cache))
    n = len(views)
    log(f"self-calibrated frames {sum(v.cal is not None for v in views)}/{n}")
    pairs = [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 1 + NEIGHBOURS))]
    pairs += [(j, i) for i, j in pairs]
    C, c, comp_of, comps, edges = layout(views, pairs, log)
    pdir = os.path.join(work, "pseudo")
    df, meta = write_pseudo(views, C, c, comp_of, pdir, "video", fps=fps)
    plan, dbg = run_geometry(pdir, stride=1, drift="on")
    plan["meta"].update({"frames_total": n, "frames_placed": int(len(df)),
                         "frames_in_main_component": int(df["main_component"].sum()),
                         "components": [len(cc) for cc in comps if len(cc) > 1][:20],
                         "registered_pairs": len(edges), "upright_rotation_deg": rot})
    return plan, dbg
