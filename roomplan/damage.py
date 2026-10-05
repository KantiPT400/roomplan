"""Per-surface damage regions, concealed-damage flags.

No damage-labelled data was available (the sample captures show clean walls), so this is a transparent
rule-based detector, not a trained model. It is built to fail quiet rather than loud:

1. Geometry first. Only pixels whose back-projected 3D point lies on a measured wall plane (within 3 cm,
   inside its span, between floor and ceiling) are examined. Furniture, doors and objects in front of a
   wall are excluded by geometry, not by appearance.
2. Appearance relative to the same wall. CIE-Lab lightness and yellow-blue (b*) residuals against a large
   median background of that wall; thresholds are in robust sigmas of the wall's own texture.
3. Multi-view voting on the wall surface. Every wall has a 2 cm (u = along wall, v = height) grid; a cell is
   damaged only if >= 2 frames AND >= 40% of the frames that saw it flag it. Shadows, glare and reflections
   move between views and are voted out.
4. Regions are connected components of flagged cells, so their extent is metric (m^2 on the wall plane).
   Class rules: thin & elongated -> crack; brighter or off-plane -> surface_loss; dark with many speckles ->
   mould_like; otherwise (dark and/or yellow-brown) -> stain_discoloration.

Concealed-damage flags record the rule that fired (see RULES).
"""
from __future__ import annotations
import os
import subprocess
import numpy as np
import cv2
from scipy import ndimage
from .uncertainty import ci, TIER_SYSTEMATIC

CELL = 0.02
RULES = {
    "ceiling_junction": "stain/mould whose top edge is within 0.25 m of the ceiling: possible leak above "
                        "(roof, upstairs bathroom, AC line)",
    "floor_junction": "stain/mould whose bottom edge is within 0.25 m of the floor: possible pipe leak in "
                      "the wall or rising damp behind the skirting",
    "both_faces": "damage on both faces of the same wall (parallel planes 5-30 cm apart): moisture through "
                  "the wall",
    "plane_bulge": "wall surface off its fitted plane by > 15 mm (mean over >= 4 views) across >= 0.1 m^2: "
                   "hidden swelling or detached plaster",
}


def _frames_lidar(scan, every_s=1.0, max_n=150):
    t = scan.poses["timestamp"].to_numpy(); f = scan.poses["frame"].to_numpy()
    out, last = [], -1e9
    have = set(int(x[:-4]) for x in os.listdir(os.path.join(scan.root, "depth")) if x.endswith(".png"))
    for ti, fi in zip(t, f):
        if ti - last >= every_s and int(fi) in have:
            out.append(int(fi)); last = ti
    if len(out) > max_n:
        out = out[:: int(np.ceil(len(out) / max_n))]
    return out


def _decode(video, frames, out_dir, width=960):
    os.makedirs(out_dir, exist_ok=True)
    need = [i for i in frames if not os.path.exists(os.path.join(out_dir, f"{i:06d}.jpg"))]
    if need:
        expr = "+".join("eq(n\\,%d)" % i for i in need)
        subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", f"select={expr},scale={width}:-2",
                        "-vsync", "0", "-q:v", "2", os.path.join(out_dir, "o_%06d.jpg")], check=True)
        for k, i in enumerate(need):
            src = os.path.join(out_dir, "o_%06d.jpg" % (k + 1))
            if os.path.exists(src):
                os.replace(src, os.path.join(out_dir, f"{i:06d}.jpg"))
    return {i: os.path.join(out_dir, f"{i:06d}.jpg") for i in frames
            if os.path.exists(os.path.join(out_dir, f"{i:06d}.jpg"))}


def _wall_frame(w, R2):
    """origin (world xz), unit along-wall vector (world xz), unit normal (world xz) for a Manhattan wall."""
    if w.axis == "x":
        p0 = np.array([w.coord, w.lo]); along = np.array([0.0, 1.0]); nrm = np.array([1.0, 0.0])
    else:
        p0 = np.array([w.lo, w.coord]); along = np.array([1.0, 0.0]); nrm = np.array([0.0, 1.0])
    # rotated -> world: world = rot @ R2
    return p0 @ R2, along @ R2, nrm @ R2


def _classify(cells_dark, cells_bright, cells_yellow, comp, offplane):
    ys, xs = np.nonzero(comp)
    n = len(ys)
    if n == 0:
        return None
    cov = np.cov(np.stack([xs, ys])) if n > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    elong = np.sqrt(max(ev[0], 1e-6) / max(ev[1], 1e-6))
    minor = 4 * np.sqrt(max(ev[1], 1e-6)) * CELL
    major = 4 * np.sqrt(max(ev[0], 1e-6)) * CELL
    if elong > 6 and minor < 0.06 and major >= 0.30:
        return "crack"
    if (cells_bright & comp).sum() > 0.5 * n or (offplane & comp).sum() > 0.3 * n:
        return "surface_loss"
    lab, k = ndimage.label(cells_dark & comp)
    if k >= 5 and (cells_dark & comp).sum() < 0.7 * comp.sum():
        return "mould_like"
    return "stain_discoloration"


def detect(plan, dbg, capture_dir, tier, out_dir, wall_tol=0.03, min_len=0.8):
    scan = dbg.get("scan"); walls = dbg.get("walls"); R2 = dbg.get("R2"); fy = dbg.get("floor_y")
    if scan is None or not walls:
        return [], []
    from .io import read_depth, read_meta
    meta = read_meta(scan.root)
    work = os.path.join(out_dir, "work", "damage")
    if tier == "lidar":
        frames = _frames_lidar(scan)
        video = os.path.join(capture_dir, "rgb.mp4")
        if not os.path.exists(video):
            return [], []
        fdir = os.environ.get("ROOMPLAN_FRAMES_DIR")   # evaluation hook: pre-rendered frames (inject_damage.py)
        imgs = (_decode(video, frames, os.path.join(work, "frames")) if not fdir else
                {i: os.path.join(fdir, f"{i:06d}.jpg") for i in frames if os.path.exists(os.path.join(fdir, f"{i:06d}.jpg"))})
    else:
        img_dir = os.path.join(os.path.dirname(scan.root), "images")
        imgs = {int(r.frame): os.path.join(img_dir, r.image) for r in scan.poses.itertuples()
                if isinstance(getattr(r, "image", None), str)}
        frames = sorted(imgs)
    ceil_y = {r["id"]: (r["ceiling_height_m"]["value"] or None) for r in plan["rooms"]}
    top_default = fy + 2.6
    W = [w for w in walls if w.length >= min_len]
    grids = []
    for w in W:
        nu = int(np.ceil(w.length / CELL)) + 1; nv = int(np.ceil((top_default - fy) / CELL)) + 1
        G = {k: np.zeros((nv, nu), np.int16) for k in ("seen", "dark", "bright", "yellow", "off")}
        G["rgb"] = np.zeros((nv, nu, 3), np.float32)
        G["dev"] = np.zeros((nv, nu), np.float32)     # summed signed deviation from the plane (m)
        grids.append(G)
    frames_used = 0
    for f in frames:
        if f not in imgs:
            continue
        img = cv2.imread(imgs[f])
        if img is None:
            continue
        d = read_depth(scan, f)
        h, w_ = img.shape[:2]
        dz = cv2.resize(d, (w_, h), interpolation=cv2.INTER_NEAREST)
        K = scan.K_depth_for(f, d.shape).copy(); K[:2] *= w_ / d.shape[1]
        R, t = scan.pose(f)
        step = 2
        v, u = np.mgrid[0:h:step, 0:w_:step]
        z = dz[::step, ::step]
        ok = (z > 0.3) & (z < 4.0)
        P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
        Pw = P @ R.T + t
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[::step, ::step, 0]; bb = lab[::step, ::step, 2]
        frames_used += 1
        for wi, (w, G) in enumerate(zip(W, grids)):
            p0, along, nrm = _wall_frame(w, R2)
            rel = Pw[..., [0, 2]] - p0
            dist = rel @ nrm; s = rel @ along; hgt = Pw[..., 1] - fy
            m = ok & (np.abs(dist) < wall_tol) & (s > 0.05) & (s < w.length - 0.05) & (hgt > 0.08) & \
                (hgt < top_default - fy - 0.05)
            if m.sum() < 400:
                continue
            m8 = m.astype(np.uint8)
            m_in = cv2.erode(m8, np.ones((7, 7), np.uint8)) > 0
            if m_in.sum() < 200:
                continue
            # background lighting: robust plane L ~ a + b*u + c*v over this wall's pixels in this frame.
            # (A median filter smaller than the stain follows the stain and cancels it.)
            Lb = _robust_plane(L, u, v, m_in)
            rL = L - Lb
            mad = 1.4826 * np.median(np.abs(rL[m_in] - np.median(rL[m_in]))) + 1.0
            bmed = np.median(bb[m_in]); bmad = 1.4826 * np.median(np.abs(bb[m_in] - bmed)) + 1.0
            dark = m_in & (rL < -max(2.5 * mad, 8))
            bright = m_in & (rL > max(5 * mad, 14))
            yellow = m_in & (bb - bmed > max(3 * bmad, 5))
            off = m_in & (np.abs(dist) > 0.008) & (np.abs(dist) < wall_tol)
            iu = np.clip((s / CELL).astype(int), 0, G["seen"].shape[1] - 1)
            iv = np.clip((hgt / CELL).astype(int), 0, G["seen"].shape[0] - 1)
            for key, mask in (("seen", m_in), ("dark", dark), ("bright", bright), ("yellow", yellow),
                              ("off", off)):
                cell = np.zeros_like(G[key], bool)
                cell[iv[mask], iu[mask]] = True
                G[key] += cell
            # wall elevation texture (mean colour per cell) and mean signed plane deviation per cell
            col = img[::step, ::step][m_in].astype(np.float32)
            np.add.at(G["rgb"], (iv[m_in], iu[m_in]), col / 1.0)
            cnt = np.zeros(G["seen"].shape, np.float32); np.add.at(cnt, (iv[m_in], iu[m_in]), 1.0)
            G.setdefault("npx", np.zeros(G["seen"].shape, np.float32))
            G["npx"] += cnt
            np.add.at(G["dev"], (iv[m_in], iu[m_in]), dist[m_in])
            G.setdefault("res", np.zeros(G["seen"].shape, np.float32))
            np.add.at(G["res"], (iv[m_in], iu[m_in]), (rL / mad)[m_in])   # lightness residual in wall sigmas
            G.setdefault("bres", np.zeros(G["seen"].shape, np.float32))
            np.add.at(G["bres"], (iv[m_in], iu[m_in]), ((bb - bmed) / bmad)[m_in])   # yellow residual
    rel_sys = TIER_SYSTEMATIC[tier][1]
    if meta.get("scale_sigma_rel"):
        rel_sys = float(np.hypot(rel_sys, meta["scale_sigma_rel"]))
    out, flags = [], []
    k = 0
    rejected = {"small": 0, "neutral_dark": 0}
    for w, G in zip(W, grids):
        seen = G["seen"]
        if seen.max() == 0:
            continue
        frac = lambda key: (G[key] >= 3) & (G[key] >= 0.6 * np.maximum(seen, 1))
        D, B, Y = frac("dark"), frac("bright"), frac("yellow")
        npx = np.maximum(G.get("npx", np.ones_like(seen, np.float32)), 1)
        mdev = G["dev"] / npx
        # off-plane: mean deviation beyond 15 mm (iPhone LiDAR noise is ~1 cm per pixel)
        O = (np.abs(mdev) > 0.015) & (seen >= 4)
        # a bright patch is only "surface loss" when the surface is also off-plane; bright alone is a
        # switch plate, glare or tile
        anyd = D | Y | (B & ndimage.binary_dilation(O, iterations=2))
        # bridge gaps of up to 3 cells along horizontal/vertical lines (cracks break up into dashes),
        # then the usual 1-cell closing for blobs
        anyd = anyd | ndimage.binary_closing(anyd, structure=np.ones((1, 4), bool)) | \
            ndimage.binary_closing(anyd, structure=np.ones((4, 1), bool))
        anyd = ndimage.binary_closing(anyd, iterations=1)
        lab, n = ndimage.label(anyd)
        surface = _surface_id(plan, w)
        for i in range(1, n + 1):
            comp = lab == i
            area = comp.sum() * CELL * CELL
            cls = _classify(D, B, Y, comp, O)
            if cls is None:
                continue
            # size: stains/mould/surface loss under 0.04 m^2 (20x20 cm) are indistinguishable from handles,
            # switch plates and sockets at this resolution; cracks are judged by length instead
            if cls != "crack" and area < 0.04:
                rejected["small"] += 1
                continue
            # a crack is a thin dark line INSIDE one surface: the wall must look the same on both sides of
            # it. The edge of a counter, shelf or skirting has different brightness on each side.
            if cls == "crack":
                mres = G["res"] / npx
                ys_, xs_ = np.nonzero(comp)
                horiz = np.ptp(xs_) >= np.ptp(ys_)
                sh = (4, 0) if horiz else (0, 4)
                side_a = ndimage.shift(comp.astype(float), sh, order=0) > 0.5
                side_b = ndimage.shift(comp.astype(float), (-sh[0], -sh[1]), order=0) > 0.5
                ok_a, ok_b = side_a & (seen > 0) & ~comp, side_b & (seen > 0) & ~comp
                ma_, mb_ = (mres[ok_a].mean(), mres[ok_b].mean()) if ok_a.sum() >= 5 and ok_b.sum() >= 5 else (0, 9)
                if ok_a.sum() < 5 or ok_b.sum() < 5 or abs(ma_ - mb_) > 0.6 or mres[comp].mean() > min(ma_, mb_) - 2.0:
                    rejected["edge_not_crack"] = rejected.get("edge_not_crack", 0) + 1
                    continue
            # colour: water stains have a yellow-brown cast (tidemarks); a shadow under furniture, a handle or
            # a switch is neutral dark. Neutral-dark regions are only kept when speckled (mould_like).
            # (An edge-softness rule was tried first and rejected the staged stain while keeping a shadow.)
            if cls == "stain_discoloration" and (Y & comp).sum() < 0.3 * comp.sum():
                rejected["neutral_dark"] = rejected.get("neutral_dark", 0) + 1
                continue
            vs, us = np.nonzero(comp)
            per = (comp ^ ndimage.binary_erosion(comp)).sum() * CELL
            sA = float(np.hypot(0.5 * per * CELL, 2 * rel_sys * area))
            k += 1
            d = {"id": f"dmg{k}", "surface": surface, "class": cls, "extent_m2": ci(area, sA, 4),
                 "bbox_on_surface_m": [round(us.min() * CELL, 2), round(vs.min() * CELL, 2),
                                       round((us.max() + 1) * CELL, 2), round((vs.max() + 1) * CELL, 2)],
                 "score": round(float(np.mean(np.maximum(G['dark'], G['yellow'])[comp] /
                                              np.maximum(seen[comp], 1))), 3),
                 "views": int(np.median(seen[comp]))}
            if cls == "crack":
                d["length_m"] = ci((us.max() - us.min() + vs.max() - vs.min() + 1) * CELL, 2 * CELL, 3)
            out.append(d)
            top = (vs.max() + 1) * CELL; bot = vs.min() * CELL
            cy = _ceiling_over(plan, w, fy)
            if cls in ("stain_discoloration", "mould_like"):
                if cy is not None and cy - top <= 0.25:
                    flags.append({"surface": surface, "rule": "ceiling_junction", "evidence": d["id"],
                                  "description": RULES["ceiling_junction"]})
                if bot <= 0.25:
                    flags.append({"surface": surface, "rule": "floor_junction", "evidence": d["id"],
                                  "description": RULES["floor_junction"]})
        # bulge rule (LiDAR only: mono depth is not flat enough to trust 8 mm)
        if tier == "lidar":
            ol, on = ndimage.label(O)
            for i in range(1, on + 1):
                if (ol == i).sum() * CELL * CELL >= 0.10:
                    flags.append({"surface": surface, "rule": "plane_bulge", "evidence": f"{surface}:off-plane",
                                  "description": RULES["plane_bulge"]})
                    break
    _elevations(W, grids, out, plan, os.path.join(out_dir, "walls"))
    if os.environ.get("ROOMPLAN_DAMAGE_DEBUG"):
        np.savez_compressed(os.path.join(out_dir, "damage_grids.npz"),
                            **{f"{w.axis}{w.coord:+.3f}_{k}": G[k] for w, G in zip(W, grids) for k in G})
    # both faces of one wall
    byw = {}
    for d in out:
        byw.setdefault(d["surface"], []).append(d)
    for i, a in enumerate(W):
        for b in W[i + 1:]:
            if a.axis == b.axis and 0.05 < abs(a.coord - b.coord) < 0.30 and \
                    min(a.hi, b.hi) - max(a.lo, b.lo) > 0.3:
                sa, sb = _surface_id(plan, a), _surface_id(plan, b)
                if byw.get(sa) and byw.get(sb):
                    flags.append({"surface": sa, "rule": "both_faces", "evidence": f"{sa}+{sb}",
                                  "description": RULES["both_faces"]})
    for d in out:
        d.setdefault("room", d["surface"].split("w")[0] if d["surface"].startswith("r") else None)
    plan["meta"]["damage_detector"] = {"frames": frames_used, "walls": len(W), "rejected": rejected,
                                       "elevations": os.path.join(out_dir, "walls")}
    return out, flags


def _robust_plane(L, u, v, mask, iters=4):
    uu, vv, ll = u[mask].astype(np.float64), v[mask].astype(np.float64), L[mask].astype(np.float64)
    A = np.stack([np.ones_like(uu), uu, vv], 1)
    wgt = np.ones_like(ll)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A * wgt[:, None], ll * wgt, rcond=None)
        r = ll - A @ coef
        sc = 1.4826 * np.median(np.abs(r)) + 1e-3
        wgt = 1.0 / np.maximum(1.0, np.abs(r) / (2.0 * sc))
    return (coef[0] + coef[1] * u + coef[2] * v).astype(np.float32)


def _surface_id(plan, w):
    best = None
    for r in plan["rooms"]:
        for e in r["walls"]:
            a, b = np.array(e["from"]), np.array(e["to"])
            ax = "x" if abs(a[0] - b[0]) < abs(a[1] - b[1]) else "z"
            if ax != w.axis:
                continue
            c = a[0] if ax == "x" else a[1]
            span = sorted((a[1], b[1])) if ax == "x" else sorted((a[0], b[0]))
            ov = min(span[1], w.hi) - max(span[0], w.lo)
            if abs(c - w.coord) < 0.06 and ov > 0.2 and (best is None or ov > best[0]):
                best = (ov, e["id"])
    return best[1] if best else f"wall_{w.axis}{w.coord:+.2f}"


def _ceiling_over(plan, w, fy):
    hs = [r["ceiling_height_m"]["value"] for r in plan["rooms"] if r["ceiling_height_m"]["value"]]
    return float(np.median(hs)) if hs else None


def _elevations(W, grids, dets, plan, out_dir):
    """Save each wall as an orthographic elevation image (2 cm/px) with damage boxes: a reviewable record."""
    os.makedirs(out_dir, exist_ok=True)
    for w, G in zip(W, grids):
        npx = G.get("npx")
        if npx is None or npx.max() == 0:
            continue
        sid = _surface_id(plan, w)
        img = (G["rgb"] / np.maximum(npx, 1)[..., None]).astype(np.uint8)
        img[npx == 0] = (40, 40, 40)
        img = cv2.resize(img[::-1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        H = img.shape[0]
        for d in dets:
            if d["surface"] != sid:
                continue
            u0, v0, u1, v1 = d["bbox_on_surface_m"]
            p0 = (int(u0 / CELL * 4), int(H - v1 / CELL * 4)); p1 = (int(u1 / CELL * 4), int(H - v0 / CELL * 4))
            cv2.rectangle(img, p0, p1, (0, 0, 255), 2)
            cv2.putText(img, f"{d['id']} {d['class'][:5]}", (p0[0], max(p0[1] - 4, 10)), 0, 0.4, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(out_dir, f"{sid}.jpg"), img)
