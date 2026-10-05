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
import json
import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation
from .mono import disparity
from .singleview import calibrate, depth_from
from .pseudo import CAM_HEIGHT_PRIOR
from .lidar import run as run_geometry

EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
WIDTH = 960          # long side is scaled so the SHORT side is 960 px? no: width (x) is 960 px
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


def _register(a, b, kps, des, K, depth, min_inl=25):
    """Pose of camera b relative to camera a (X_b = R X_a + t, metres in a's depth scale)."""
    if des[a] is None or des[b] is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des[a], des[b], k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.78 * x[1].distance]
    if len(good) < min_inl:
        return None
    pa = np.float32([kps[a][g.queryIdx].pt for g in good])
    pb = np.float32([kps[b][g.trainIdx].pt for g in good])
    z = depth[a]
    if z is None:
        return None
    u = np.clip((pa[:, 0] / DS).astype(int), 0, z.shape[1] - 1)
    v = np.clip((pa[:, 1] / DS).astype(int), 0, z.shape[0] - 1)
    zz = z[v, u]
    Ka = K[a]
    X = np.stack([(pa[:, 0] - Ka[0, 2]) / Ka[0, 0] * zz, (pa[:, 1] - Ka[1, 2]) / Ka[1, 1] * zz, zz], 1)
    ok = (zz > 0.2) & (zz < 8)
    if ok.sum() < min_inl:
        return None
    r = cv2.solvePnPRansac(X[ok].astype(np.float64), pb[ok].astype(np.float64), K[b], None,
                           iterationsCount=1000, reprojectionError=6.0, confidence=0.999,
                           flags=cv2.SOLVEPNP_EPNP)
    if not r[0] or r[3] is None or len(r[3]) < min_inl:
        return None
    inl = r[3][:, 0]
    rv, tv = cv2.solvePnPRefineLM(X[ok][inl], pb[ok][inl].astype(np.float64), K[b], None, r[1], r[2])
    R, _ = cv2.Rodrigues(rv)
    if not np.all(np.isfinite(tv)) or np.linalg.norm(tv) > 6.0:
        return None
    # depth-scale ratio: b's own depth at the inliers vs depth predicted from a's points
    Xb = X[ok][inl] @ R.T + tv[:, 0]
    return {"R": R, "t": tv[:, 0], "inliers": int(len(inl)), "ratio": float(len(inl) / ok.sum()),
            "pts_b": pb[ok][inl], "zb_pred": Xb[:, 2]}


def _reuse(out_dir):
    """--reuse: geometry only, on the pseudo-LiDAR folder a previous run of this capture already wrote."""
    import json as _j
    p = os.path.join(out_dir, "work", "pseudo")
    return p if os.environ.get("ROOMPLAN_REUSE") and os.path.exists(os.path.join(p, "meta.json")) else None


def run(capture, out_dir, prior=CAM_HEIGHT_PRIOR, log=print):
    from .mvreg import View, layout, write_pseudo
    if _reuse(out_dir):
        plan, dbg = run_geometry(_reuse(out_dir), stride=1, drift="off")
        plan["meta"]["reused_registration"] = True
        return plan, dbg
    work = os.path.join(out_dir, "work")
    img_dir = os.path.join(work, "images")
    items = collect(capture, img_dir)
    cache = os.path.join(work, "disp_cache")
    os.makedirs(cache, exist_ok=True)
    views = [View(it["name"], cv2.imread(os.path.join(img_dir, it["name"])), _K(it), it["room"], cache)
             for it in items]
    n = len(views)
    log(f"photos {n}, self-calibrated {sum(v.cal is not None for v in views)}")
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    pdir = os.path.join(work, "pseudo")
    try:
        C, c, comp_of, comps, edges = layout(views, pairs, log)
        df, meta = write_pseudo(views, C, c, comp_of, pdir, "photo", prior=prior)
        plan, dbg = run_geometry(pdir, stride=1, drift="off")
    except Exception as e:                          # too few usable photos to map anything
        log(f"stitched mapping failed ({e!r}); single-view estimates only")
        plan = {"meta": {"tier": "photo", "mapping_error": repr(e)}, "rooms": [], "openings": [], "adjacency": []}
        dbg, edges, comps = {}, [], []
        df = pd.DataFrame(columns=["x", "z", "group", "main_component", "image"])
    from matplotlib.path import Path
    if len(df) and "R2" in dbg:
        cam = df[["x", "z"]].to_numpy() @ dbg["R2"].T
        for r in plan["rooms"]:
            inside = Path(np.array(r["polygon_m"])).contains_points(cam)
            if inside.any():
                vals, cnt = np.unique(df["group"].to_numpy()[inside], return_counts=True)
                r["name"] = str(vals[np.argmax(cnt)])
                r["photo_folders"] = {str(v): int(k) for v, k in zip(vals, cnt)}
            r["stitched"] = bool(inside.any() and df["main_component"].to_numpy()[inside].mean() > 0.5)
    folders = sorted(set(it["room"] for it in items))
    main_f = sorted(set(df.loc[df["main_component"] == 1, "group"])) if len(df) else []
    # any picture in, results out: folders with no stitched room get a single-view estimate beside the plan
    named = {r.get("name") for r in plan["rooms"]}
    missing = [f for f in folders if f not in named]
    if missing:
        from .fallback import estimate_rooms
        xs = [p[0] for r in plan["rooms"] for p in r["polygon_m"]]
        x0 = (max(xs) + 2.0) if xs else 0.0
        plan["rooms"] += estimate_rooms(views, missing, x0)
    plan["meta"].update({
        "photos_total": n, "photos_self_calibrated": int(sum(v.cal is not None for v in views)),
        "photos_placed": int(len(df)), "registered_pairs": len(edges),
        "components": [len(cc) for cc in comps if len(cc) > 1],
        "unregistered_photos": sorted(set(v.name for v in views) - set(df["image"])),
        "room_folders_stitched": main_f,
        "room_folders_unstitched": sorted(set(folders) - set(main_f)),
    })
    return plan, dbg
