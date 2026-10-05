"""Turn SfM fragments + monocular depth into a metric "pseudo-LiDAR" capture that the LiDAR-tier
geometry runs on unchanged (same walls/rooms/openings code, same output contract).

Per image : relative disparity d from the mono model is mapped to depth with 1/z = a*d + b, (a, b)
            fitted robustly to the depths of that image's SfM points.
Per fragment: gravity = normal of the floor plane, chosen among RANSAC planes as the one the cameras
            stay at a near-constant height above AND mostly look towards (ceilings are also at constant
            height but are rarely looked at). Metric scale = camera-height prior / median camera height.
Across fragments (video): walls give yaw modulo 90 deg; the remaining 4-way ambiguity and the offset are
            resolved by continuity of heading and position across the time gap, and the residual
            misalignment is left to the drift pose graph (fragment starts are passed as pose breaks).
"""
from __future__ import annotations
import json
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from .mono import disparity_cached
from .planes import dominant_angle

# Camera height above floor when filming handheld at chest height. Calibrated on the three sample
# LiDAR captures (medians 1.406, 1.460 m; same operator). The sigma covers operator-to-operator
# spread and is the dominant term of the video/photo error budget.
CAM_HEIGHT_PRIOR = (1.40, 0.07)

DW, DH = 256, 192


def align_disparity(d, K_s, uv_s, zs, min_pts=15):
    u = np.clip(uv_s[:, 0].round().astype(int), 0, d.shape[1] - 1)
    v = np.clip(uv_s[:, 1].round().astype(int), 0, d.shape[0] - 1)
    x = d[v, u]
    y = 1.0 / np.maximum(zs, 1e-6)
    ok = (zs > 0) & np.isfinite(x)
    x, y = x[ok], y[ok]
    if len(x) < min_pts:
        return None
    w = np.ones_like(x)
    a = b = 0.0
    for _ in range(8):
        A = np.stack([x, np.ones_like(x)], 1) * w[:, None]
        sol, *_ = np.linalg.lstsq(A, y * w, rcond=None)
        a, b = sol
        r = (a * x + b - y) / np.maximum(y, 1e-6)
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = 1.0 / np.maximum(1.0, np.abs(r) / (2.5 * s))
    if a <= 0:
        return None
    inl = np.abs((a * x + b - y) / y) < 0.15
    return float(a), float(b), float(np.median(np.abs((a * x + b - y) / y))), int(inl.sum())


def fragment_depths(frag, image_dir, cache, max_rel_err=0.12):
    s = DW / frag.size[0]
    K_s = frag.K.copy(); K_s[:2] *= s
    out = {}
    for n in frag.names:
        uv, X = frag.obs[n]
        if len(uv) < 15:
            continue
        Xc = X @ frag.R[n].T + frag.t[n]
        d = disparity_cached(os.path.join(image_dir, n), cache, (DW, int(round(frag.size[1] * s))))
        al = align_disparity(d, K_s, uv * s, Xc[:, 2])
        if al is None or al[2] > max_rel_err:
            continue
        a, b, err, ninl = al
        den = a * d + b
        z = np.where(den > 1e-6, 1.0 / np.maximum(den, 1e-6), 0.0)
        out[n] = (z.astype(np.float32), err)
    return out, K_s


def _points(frag, depths, K_s, step=6):
    pts = []
    for n, (z, _) in depths.items():
        h, w = z.shape
        v, u = np.mgrid[0:h:step, 0:w:step]
        zz = z[0:h:step, 0:w:step]
        m = zz > 0
        P = np.stack([(u[m] - K_s[0, 2]) / K_s[0, 0] * zz[m], (v[m] - K_s[1, 2]) / K_s[1, 1] * zz[m], zz[m]], 1)
        pts.append((P - frag.t[n]) @ frag.R[n])     # camera -> fragment world
    return np.concatenate(pts)


def _ransac_planes(P, thr, n_planes=6, iters=400, min_frac=0.04, rng=None):
    rng = rng or np.random.default_rng(0)
    rest = P.copy()
    planes = []
    for _ in range(n_planes):
        if len(rest) < 200:
            break
        best, bn = None, 0
        for _ in range(iters):
            s = rest[rng.choice(len(rest), 3, replace=False)]
            nrm = np.cross(s[1] - s[0], s[2] - s[0])
            L = np.linalg.norm(nrm)
            if L < 1e-9:
                continue
            nrm /= L
            dd = -nrm @ s[0]
            inl = np.abs(rest @ nrm + dd) < thr
            k = inl.sum()
            if k > bn:
                best, bn = (nrm, dd, inl), k
        if best is None or bn < min_frac * len(P):
            break
        nrm, dd, inl = best
        Q = rest[inl]
        c = Q.mean(0)
        nrm = np.linalg.svd(Q - c)[2][-1]
        planes.append((nrm, -nrm @ c, len(Q)))
        rest = rest[~inl]
    return planes


def gravity_and_scale(frag, depths, K_s, prior=CAM_HEIGHT_PRIOR):
    P = _points(frag, depths, K_s)
    C = np.array([frag.center(n) for n in depths])
    F = np.array([frag.R[n].T @ np.array([0, 0, 1.0]) for n in depths])   # optical axes in world
    ext = np.percentile(np.linalg.norm(P - np.median(P, 0), axis=1), 90)
    planes = _ransac_planes(P, thr=0.01 * ext)
    best = None
    for nrm, dd, cnt in planes:
        dist = C @ nrm + dd
        side = np.sign(np.median(dist))
        if side == 0 or np.mean(np.sign(dist) == side) < 0.9:
            continue
        up = nrm * side                     # from plane towards the cameras
        h = np.abs(dist)
        cv = h.std() / max(h.mean(), 1e-9)
        look = float(np.mean(F @ (-up)))    # >0: cameras look towards this plane
        if cv > 0.35:
            continue
        score = look + 0.2 * cnt / len(P) - cv
        if best is None or score > best[0]:
            best = (score, up, np.median(h), cv, look)
    if best is None:
        return None
    _, up, hmed, cv, look = best
    s = prior[0] / hmed
    # rotation taking `up` to +y
    Rg, _ = Rotation.align_vectors([[0, 1, 0]], [up])
    return {"R": Rg.as_matrix(), "scale": float(s), "cam_height_units": float(hmed),
            "height_cv": float(cv), "look": float(look)}


def _cam_to_world_pose(frag, n, G, yaw_R, offset):
    """camera-to-world rotation and position after gravity, scale, yaw and offset."""
    Rwc = frag.R[n].T
    c = frag.center(n)
    Rtot = yaw_R @ G["R"]
    return Rtot @ Rwc, Rtot @ (c * G["scale"]) + offset


def build(frags, image_dir, out_dir, cache, tier, fps=None, prior=CAM_HEIGHT_PRIOR, log=print):
    """Merge fragments and write a Record3D-style folder. Returns a summary dict."""
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    prepared = []
    for fi, fr in enumerate(frags):
        depths, K_s = fragment_depths(fr, image_dir, cache)
        if len(depths) < 3:
            log(f"fragment {fi}: {len(depths)} usable frames, dropped")
            continue
        G = gravity_and_scale(fr, depths, K_s, prior)
        if G is None:
            log(f"fragment {fi}: no floor plane, dropped")
            continue
        # Manhattan yaw in the gravity-aligned, scaled frame
        P = (_points(fr, depths, K_s) * G["scale"]) @ G["R"].T
        floor = np.percentile(P[:, 1], 3)
        try:
            th = dominant_angle(P, floor + 0.3, floor + 2.0)
        except Exception:
            th = 0.0
        prepared.append(dict(frag=fr, depths=depths, K_s=K_s, G=G, theta=th, floor=floor))
        log(f"fragment {fi}: {len(depths)}/{len(fr.names)} frames, scale {G['scale']:.3f}, "
            f"cam height cv {G['height_cv']:.2f}, manhattan {th:.1f} deg")
    if not prepared:
        raise RuntimeError("no usable SfM fragment")
    # chain fragments in time order
    poses = {}
    breaks = []
    prev = None
    for k, pr in enumerate(prepared):
        fr, G = pr["frag"], pr["G"]
        base = Rotation.from_euler("y", pr["theta"], degrees=True).as_matrix()
        if prev is None:
            yaw_R, off = base, np.zeros(3)
        else:
            # heading/position continuity with the previous fragment's last frame
            pf, pG, pyaw, poff = prev
            ln = [n for n in pf.names if n in prev_depths][-1]
            Rl, cl = _cam_to_world_pose(pf, ln, pG, pyaw, poff)
            fn = [n for n in fr.names if n in pr["depths"]][0]
            best = None
            for q in range(4):
                cand = Rotation.from_euler("y", 90 * q, degrees=True).as_matrix() @ base
                Rf, cf = _cam_to_world_pose(fr, fn, G, cand, np.zeros(3))
                ang = np.degrees(np.arccos(np.clip((Rl[:, 2] * np.array([1, 0, 1])) @
                                                   (Rf[:, 2] * np.array([1, 0, 1])) /
                                                   (np.linalg.norm(Rl[:, 2][[0, 2]]) *
                                                    np.linalg.norm(Rf[:, 2][[0, 2]]) + 1e-9), -1, 1)))
                if best is None or ang < best[0]:
                    best = (ang, cand, cf)
            _, yaw_R, cf = best
            off = cl - cf
            off[1] = 0.0
            breaks.append(fn)
        prev = (fr, G, yaw_R, off)
        prev_depths = pr["depths"]
        for n in fr.names:
            if n in pr["depths"]:
                Rwc, c = _cam_to_world_pose(fr, n, G, yaw_R, off)
                poses[n] = (Rwc, c, pr["depths"][n][0] * G["scale"], pr["K_s"], k)
    # write folder
    names = sorted(poses)
    rows = []
    K_s = prepared[0]["K_s"]
    dh = poses[names[0]][2].shape[0]
    for i, n in enumerate(names):
        Rwc, c, z, Ks, k = poses[n]
        cv2.imwrite(os.path.join(out_dir, "depth", f"{i:06d}.png"),
                    np.clip(z * 1000, 0, 65535).astype(np.uint16))
        q = Rotation.from_matrix(Rwc).as_quat()
        ts = (i / fps) if fps else float(i)
        rows.append([ts, i, *c, *q, Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2], n, k])
    import pandas as pd
    df = pd.DataFrame(rows, columns=["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw",
                                     "fx", "fy", "cx", "cy", "image", "fragment"])
    df.to_csv(os.path.join(out_dir, "odometry.csv"), index=False)
    # intrinsics expressed for a 1920-wide image, matching the LiDAR folder layout
    sc = 1920 / DW
    np.savetxt(os.path.join(out_dir, "camera_matrix.csv"),
               np.array([[K_s[0, 0] * sc, 0, K_s[0, 2] * sc], [0, K_s[1, 1] * sc, K_s[1, 2] * sc], [0, 0, 1]]),
               delimiter=",")
    br = [int(df.index[df["image"] == b][0]) for b in breaks if (df["image"] == b).any()]
    meta = {"tier": tier, "rgb_size": [1920, int(round(dh * sc))], "depth_size": [DW, dh],
            "scale_prior_m": prior, "scale_sigma_rel": prior[1] / prior[0],
            "fragments": len(prepared), "fragment_breaks": br, "frames": len(names)}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta
