"""Photo tier: one folder of 2-8 stills per room (no depth, no poses) -> one stitched plan.

All photos from all folders go into a single exhaustive-matching SfM. Rooms are tied together by the
doorway photos the capture protocol asks for (docs/capture_protocol.md, step 4): each room's folder holds a
shot taken from the doorway looking back into the room just left, which shares features with that room.
Registered photos become pseudo-LiDAR frames (pseudo.py) and the same geometry runs on them.
Room names come from folder names: each segmented room takes the name of the folder whose cameras sit
inside it. Photos that do not register are reported, never silently dropped; a folder with no registered
photo is reported as an unplaced room.
"""
from __future__ import annotations
import os
import json
import numpy as np
from PIL import Image, ImageOps
from .sfm import run_sfm
from . import pseudo
from .lidar import run as run_geometry

EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
WIDTH = 960


def _focal_35(img):
    try:
        ex = img.getexif()
        sub = ex.get_ifd(0x8769)
        f35 = sub.get(0xA405) or ex.get(0xA405)
        return float(f35) if f35 else None
    except Exception:
        return None


def collect(capture, img_dir):
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    os.makedirs(img_dir, exist_ok=True)
    rooms, f35s = {}, []
    for room in sorted(d for d in os.listdir(capture) if os.path.isdir(os.path.join(capture, d))):
        files = sorted(f for f in os.listdir(os.path.join(capture, room)) if f.lower().endswith(EXTS))
        for f in files:
            src = os.path.join(capture, room, f)
            name = f"{room}__{os.path.splitext(f)[0]}.jpg"
            dst = os.path.join(img_dir, name)
            im = Image.open(src)
            f35 = _focal_35(im)
            if f35:
                f35s.append(f35)
            if not os.path.exists(dst):
                im = ImageOps.exif_transpose(im).convert("RGB")
                h = int(round(im.height * WIDTH / im.width))
                im.resize((WIDTH, h), Image.LANCZOS).save(dst, quality=92)
            rooms.setdefault(room, []).append(name)
    return rooms, (float(np.median(f35s)) if f35s else None)


def run(capture, out_dir):
    work = os.path.join(out_dir, "work")
    img = os.path.join(work, "images")
    rooms, f35 = collect(capture, img)
    # focal from EXIF 35 mm-equivalent focal length when present (36 mm sensor width equivalent)
    focal = WIDTH * (f35 / 36.0) if f35 else 0.72 * WIDTH
    frags = run_sfm(img, os.path.join(work, "sfm"), focal_px=focal, sequential=False, min_model=2)
    pdir = os.path.join(work, "pseudo")
    logf = open(os.path.join(work, "pseudo.log"), "w")
    # photos have no time order: fragments are placed by continuity of nothing, so only the largest
    # fragment is mapped; the others are reported as unplaced (see report: failure modes)
    frags.sort(key=lambda f: -len(f.names))
    main = frags[:1]
    meta = pseudo.build(main, img, pdir, os.path.join(work, "depth_cache"), "photo", fps=None,
                        log=lambda m: (print(m), logf.write(m + "\n")))
    plan, dbg = run_geometry(pdir, stride=1, drift="off")
    # name rooms from folders
    import pandas as pd
    from matplotlib.path import Path
    od = pd.read_csv(os.path.join(pdir, "odometry.csv"))
    cam = od[["x", "z"]].to_numpy() @ dbg["R2"].T
    folder = od["image"].str.split("__").str[0].to_numpy()
    used = set()
    for r in plan["rooms"]:
        inside = Path(np.array(r["polygon_m"])).contains_points(cam)
        if inside.any():
            vals, cnt = np.unique(folder[inside], return_counts=True)
            r["name"] = str(vals[np.argmax(cnt)])
            r["photo_folders"] = {str(v): int(c) for v, c in zip(vals, cnt)}
            used.add(r["name"])
    registered = set(od["image"])
    plan["meta"]["photos_total"] = sum(len(v) for v in rooms.values())
    plan["meta"]["photos_registered"] = len(registered)
    plan["meta"]["unregistered_photos"] = sorted(set(sum(rooms.values(), [])) - registered)
    plan["meta"]["unplaced_room_folders"] = sorted(r for r, fs in rooms.items()
                                                  if not any(f in registered for f in fs))
    plan["meta"]["sfm_fragments"] = len(frags)
    plan["meta"]["focal_35mm"] = f35
    return plan, dbg
