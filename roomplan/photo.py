"""Photo tier: one folder of 2-8 stills per room (no depth, no poses) -> one stitched plan.

SfM (pycolmap) registered 9 of 45 protocol-style photos on the sample apartment (white walls, few
features), so this tier does not depend on triangulation:

1. Per photo: mono disparity -> metric, gravity-aligned depth from that photo alone (singleview.py:
   planarity fixes the disparity shift, the floor gives gravity, the camera-height prior gives scale).
   Photos that show too little floor stay uncalibrated for now.
2. Pairwise registration: SIFT matches between every pair of photos, verified by PnP-RANSAC that uses
   the metric depth of one photo and the 2D keypoints of the other. This needs ~25 good matches, not a
   triangulated track, and works across folders through the protocol's doorway shots.
3. Layout: maximum spanning tree over registration inliers; poses composed from the best-calibrated
   photo. Each camera keeps its own gravity when it has one; uncalibrated photos inherit gravity and
   depth scale from the neighbour they registered to.
4. The registered photos are written as a pseudo-LiDAR folder and the shared geometry runs on it.
   Components that never connect are reported as unplaced, never silently overlapped.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageOps
from .pseudo import CAM_HEIGHT_PRIOR
from .lidar import run as run_geometry

EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
WIDTH = 960          # every photo is resized to 960 px wide after EXIF rotation (portrait: 960 x 1280)
DS = 4               # disparity/depth stored at 1/4 resolution
DIAG_35 = 43.27      # mm, diagonal of the 36x24 frame that "35 mm equivalent" focal lengths refer to


def _f35(im):
    try:
        ex = im.getexif()
        v = ex.get_ifd(0x8769).get(0xA405) or ex.get(0xA405)
        return float(v) if v else None
    except Exception:
        return None


def collect(capture, img_dir):
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    os.makedirs(img_dir, exist_ok=True)
    items = []
    for room in sorted(d for d in os.listdir(capture) if os.path.isdir(os.path.join(capture, d))):
        for f in sorted(os.listdir(os.path.join(capture, room))):
            if not f.lower().endswith(EXTS):
                continue
            src = os.path.join(capture, room, f)
            name = f"{room}__{os.path.splitext(f)[0]}.jpg"
            dst = os.path.join(img_dir, name)
            im = Image.open(src)
            f35 = _f35(im)
            im = ImageOps.exif_transpose(im).convert("RGB")
            h = int(round(im.height * WIDTH / im.width))
            if not os.path.exists(dst):
                im.resize((WIDTH, h), Image.LANCZOS).save(dst, quality=92)
            items.append({"room": room, "name": name, "w": WIDTH, "h": h, "f35": f35})
    return items


def _K(it, default_f35=26.0 * 1.1):
    # 35 mm-equivalent focal length refers to the frame diagonal; iPhone main camera ~26 mm.
    f35 = it["f35"] or default_f35
    fpx = f35 / DIAG_35 * np.hypot(it["w"], it["h"])
    return np.array([[fpx, 0, it["w"] / 2], [0, fpx, it["h"] / 2], [0, 0, 1.0]])


def _reuse(out_dir):
    """--reuse: geometry only, on the pseudo-LiDAR folder a previous run of this capture already wrote."""
    p = os.path.join(out_dir, "work", "pseudo")
    return p if os.environ.get("ROOMPLAN_REUSE") and os.path.exists(os.path.join(p, "meta.json")) else None


def _label_rooms(plan, dbg, df):
    """Name each stitched room after the photo folder most of its photos came from. Names stay unique
    ("kitchen", "kitchen (2)" when two rooms are mostly kitchen photos); a room containing no photo position
    is "unlabelled N", so a folder name and a default name can never look alike. Returns the folders with at
    least one photo position inside a stitched room (their floor is in the plan)."""
    from matplotlib.path import Path
    seen, k_un, covered = {}, 0, set()
    cam = df[["x", "z"]].to_numpy(float) @ dbg["R2"].T if (len(df) and "R2" in dbg) else np.zeros((0, 2))
    for r in plan["rooms"]:
        inside = Path(np.array(r["polygon_m"])).contains_points(cam) if len(cam) else np.zeros(0, bool)
        if inside.any():
            vals, cnt = np.unique(df["group"].to_numpy()[inside].astype(str), return_counts=True)
            name = str(vals[np.argmax(cnt)])
            r["photo_folders"] = {str(v): int(k) for v, k in zip(vals, cnt)}
            covered |= set(vals.tolist())
        else:
            k_un += 1
            name = f"unlabelled {k_un}"
        seen[name] = seen.get(name, 0) + 1
        r["name"] = name if seen[name] == 1 else f"{name} ({seen[name]})"
        r["stitched"] = bool(inside.any() and df["main_component"].to_numpy()[inside].mean() > 0.5)
    return covered


def run(capture, out_dir, prior=CAM_HEIGHT_PRIOR, log=print):
    from .mvreg import View, layout, write_pseudo
    work = os.path.join(out_dir, "work")
    img_dir = os.path.join(work, "images")
    items = collect(capture, img_dir)
    cache = os.path.join(work, "disp_cache")
    os.makedirs(cache, exist_ok=True)
    pdir = os.path.join(work, "pseudo")
    view = lambda it: View(it["name"], cv2.imread(os.path.join(img_dir, it["name"])), _K(it), it["room"], cache)
    views, edges, comps = None, None, None
    if _reuse(out_dir):                              # --reuse: geometry only, on the previous registration
        plan, dbg = run_geometry(pdir, stride=1, drift="off")
        plan["meta"]["reused_registration"] = True
        df = pd.read_csv(os.path.join(pdir, "odometry.csv"))
    else:
        views = [view(it) for it in items]
        n = len(views)
        log(f"photos {n}, self-calibrated {sum(v.cal is not None for v in views)}")
        pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
        try:
            C, c, comp_of, comps, edges = layout(views, pairs, log)
            df, meta = write_pseudo(views, C, c, comp_of, pdir, "photo", prior=prior)
            plan, dbg = run_geometry(pdir, stride=1, drift="off")
        except Exception as e:                      # too few usable photos to map anything
            log(f"stitched mapping failed ({e!r}); single-view estimates only")
            plan = {"meta": {"tier": "photo", "mapping_error": repr(e)}, "rooms": [], "openings": [],
                    "adjacency": []}
            dbg, edges, comps = {}, [], []
            df = pd.DataFrame(columns=["x", "z", "group", "main_component", "image"])
    covered = _label_rooms(plan, dbg, df)
    folders = sorted(set(it["room"] for it in items))
    main_f = sorted(set(df.loc[df["main_component"] == 1, "group"].astype(str))) if len(df) else []
    # any picture in, results out: a folder with no photo inside any stitched room gets a single-view
    # estimate beside the plan. Coverage, not names: a folder whose photos sit in a room named after another
    # folder (merged rooms) is already in the plan, and estimating it again counted its floor twice
    # (footprint 0.96 -> 1.61 x LiDAR on the sample photo set); a folder whose one placed photo produced no
    # room is not covered and still gets its estimate.
    missing = sorted(set(folders) - covered)
    if missing:
        from .fallback import estimate_rooms
        if views is None:
            views = [view(it) for it in items if it["room"] in missing]
        xs = [p[0] for r in plan["rooms"] for p in r["polygon_m"]]
        x0 = (max(xs) + 2.0) if xs else 0.0
        plan["rooms"] += estimate_rooms(views, missing, x0)
    plan["meta"].update({
        "photos_total": len(items), "photos_placed": int(len(df)),
        "room_folders_stitched": main_f, "room_folders_unstitched": sorted(set(folders) - set(main_f)),
        "room_folders_in_rooms": sorted(covered),
        "room_folders_not_in_rooms": missing,
    })
    if views is not None and len(views) == len(items):
        plan["meta"]["photos_self_calibrated"] = int(sum(v.cal is not None for v in views))
    if edges is not None:
        plan["meta"].update({
            "registered_pairs": len(edges), "components": [len(cc) for cc in comps if len(cc) > 1],
            "unregistered_photos": sorted(set(it["name"] for it in items) - set(df["image"]))})
    return plan, dbg
