"""Pairwise multi-view registration shared by the photo and video tiers.

COLMAP's incremental SfM fragmented on the sample apartment (white walls, motion blur): 7-17 models per
video, 9 of 45 photos registered. Checked against the LiDAR capture's true poses, two-view essential-matrix
estimation is far more reliable on the same images: with >= 25 inliers the relative rotation is within
~1 deg of truth (e.g. 18.81 vs 18.92, 20.16 vs 19.88, 8.64 vs 8.96 deg). So:

  pair (a, b): SIFT + ratio test -> essential matrix (RANSAC) -> R, t direction. The translation length
               comes from a's metric single-view depth: 1-D least squares on reprojection in b.
               PnP on a's depth is the fallback when E has too few inliers.
  graph      : photos -> all pairs; video -> each frame against the next few frames.
  layout     : maximum spanning tree over inliers, then every camera keeps its own gravity (floor
               normal) where it has one; uncalibrated views inherit depth scale from their tree parent.
Output is a pseudo-LiDAR folder for the shared geometry; components that never connect are reported.
"""
from __future__ import annotations
import json
import os
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation
from .mono import disparity
from .singleview import calibrate, depth_from
from .pseudo import CAM_HEIGHT_PRIOR

DS = 4
MIN_E_INL = 25   # 40 was tried: more precise edges (50% right vs 31%) but the sparse photo graph fell apart (43 -> 8)


class View:
    def __init__(self, name, img, K, group, disp_cache):
        self.name, self.K, self.group = name, K, group
        self.h, self.w = img.shape[:2]
        cp = os.path.join(disp_cache, name + ".npy")
        if os.path.exists(cp):
            self.disp = np.load(cp)
        else:
            d = disparity(img)
            self.disp = cv2.resize(d, (self.w // DS, self.h // DS), interpolation=cv2.INTER_AREA)
            np.save(cp, self.disp)
        Ks = K.copy(); Ks[:2] /= DS
        self.cal = calibrate(self.disp, Ks)
        self.depth = depth_from(self.disp, self.cal) if self.cal else None
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Floor tiles repeat: on the sample apartment 186 of 203 essential-matrix registrations were wrong
        # (median 112 deg off the LiDAR truth), with 61% of their inliers on the floor, linking photos 4 m
        # apart. Features on the floor plane are therefore not used for matching.
        mask = None
        self.floor_mask = None
        if self.cal is not None:
            Ks = K.copy(); Ks[:2] /= DS
            z = self.depth
            hh, ww = z.shape
            vv, uu = np.mgrid[0:hh, 0:ww]
            P = np.stack([(uu - Ks[0, 2]) / Ks[0, 0] * z, (vv - Ks[1, 2]) / Ks[1, 1] * z, z], -1)
            yg = (P @ self.cal["Rg"].T)[..., 1]            # height in the gravity frame, camera at 0
            # metric scale was set so the floor plane is exactly CAM_HEIGHT_PRIOR below the camera
            fl = (yg < -CAM_HEIGHT_PRIOR[0] + 0.15)
            fl = cv2.dilate(fl.astype(np.uint8), np.ones((3, 3), np.uint8))
            self.floor_mask = cv2.resize(fl, (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0
        else:
            # no floor estimate: photos are upright and the protocol keeps floor at the bottom
            self.floor_mask = np.zeros((self.h, self.w), bool)
            self.floor_mask[int(0.65 * self.h):] = True
        mask = (~self.floor_mask).astype(np.uint8) * 255
        kp, self.des = SIFT.detectAndCompute(g, mask)
        # keep only what registration needs: keypoint positions as an array, no full-resolution mask. A 4-minute
        # clip at 6 fps is ~1400 views; KeyPoint objects + the mask cost ~2 MB per view on top of descriptors.
        self.pts = np.float32([k.pt for k in kp]).reshape(-1, 2)
        self.floor_mask = None


SIFT = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)


def _depth_at(v: View, pts):
    u = np.clip((pts[:, 0] / DS).astype(int), 0, v.depth.shape[1] - 1)
    w = np.clip((pts[:, 1] / DS).astype(int), 0, v.depth.shape[0] - 1)
    return v.depth[w, u]


def register(a: View, b: View):
    """Relative pose X_b = R X_a + t (metres, a's scale). None if unreliable."""
    if a.des is None or b.des is None or len(a.pts) < 20 or len(b.pts) < 20:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.des, b.des, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
    if len(good) < MIN_E_INL:
        return None
    pa = a.pts[[g.queryIdx for g in good]].astype(np.float64).reshape(-1, 2)
    pb = b.pts[[g.trainIdx for g in good]].astype(np.float64).reshape(-1, 2)
    na = cv2.undistortPoints(pa.reshape(-1, 1, 2), a.K, None).reshape(-1, 2)
    nb = cv2.undistortPoints(pb.reshape(-1, 1, 2), b.K, None).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                   threshold=1.5 / a.K[0, 0])
    res = None
    if E is not None and E.shape == (3, 3):
        n_inl, R, t, mask2 = cv2.recoverPose(E, na, nb, np.eye(3), mask=mask)
        inl = mask2[:, 0] > 0
        if n_inl >= MIN_E_INL and n_inl >= 0.3 * len(good):
            t = t[:, 0]
            s = 0.0
            if a.depth is not None:
                za = _depth_at(a, pa[inl])
                ok = (za > 0.2) & (za < 8)
                if ok.sum() >= 10:
                    Xa = np.stack([na[inl][ok, 0] * za[ok], na[inl][ok, 1] * za[ok], za[ok]], 1)
                    Y = Xa @ R.T
                    # project (Y + s t) and match nb: linearised 1-D least squares in s, robustified
                    obs = nb[inl][ok]
                    ss = np.linspace(0, 3.0, 301)
                    errs = []
                    for sv in ss:
                        Z = Y + sv * t
                        pr = Z[:, :2] / np.maximum(Z[:, 2:3], 1e-3)
                        errs.append(np.median(np.linalg.norm(pr - obs, axis=1)))
                    s = float(ss[int(np.argmin(errs))])
            res = {"R": R, "t": t * s, "inliers": int(n_inl), "method": "E", "scaled": a.depth is not None,
                   "pa": pa[inl], "pb": pb[inl]}
    if res is None and a.depth is not None:
        za = _depth_at(a, pa)
        ok = (za > 0.2) & (za < 8)
        if ok.sum() >= MIN_E_INL:
            X = np.stack([na[ok, 0] * za[ok], na[ok, 1] * za[ok], za[ok]], 1)
            r = cv2.solvePnPRansac(X, pb[ok], b.K, None, iterationsCount=1000, reprojectionError=6.0,
                                   flags=cv2.SOLVEPNP_EPNP)
            if r[0] and r[3] is not None and len(r[3]) >= MIN_E_INL:
                R, _ = cv2.Rodrigues(r[1])
                t = r[2][:, 0]
                if np.all(np.isfinite(t)) and np.linalg.norm(t) < 6:
                    i = r[3][:, 0]
                    res = {"R": R, "t": t, "inliers": int(len(i)), "method": "PnP", "scaled": True,
                           "pa": pa[ok][i], "pb": pb[ok][i]}
    if res is None:
        return None
    # depth consistency: a's 3-D points moved into b must land at b's own (independently estimated) depth
    if a.depth is not None and b.depth is not None and res["scaled"]:
        za = _depth_at(a, res["pa"])
        na_ = cv2.undistortPoints(res["pa"].reshape(-1, 1, 2), a.K, None).reshape(-1, 2)
        Xa = np.stack([na_[:, 0] * za, na_[:, 1] * za, za], 1)
        zb_pred = (Xa @ res["R"].T + res["t"])[:, 2]
        zb_own = _depth_at(b, res["pb"])
        ok = (za > 0.2) & (zb_own > 0.2) & (zb_pred > 0.2)
        if ok.sum() < 10:
            return None
        ratio = np.median(zb_pred[ok] / zb_own[ok])
        if not (0.7 < ratio < 1.4):
            return None
        res["depth_ratio"] = float(ratio)
    # gravity consistency between self-calibrated views
    if a.cal is not None and b.cal is not None:
        ua = a.cal["Rg"].T @ np.array([0, 1.0, 0]); ub = b.cal["Rg"].T @ np.array([0, 1.0, 0])
        if np.degrees(np.arccos(np.clip((res["R"] @ ua) @ ub, -1, 1))) > 12:
            return None
    return res


def layout(views, pairs, log=print):
    edges = []
    for a, b in pairs:
        r = register(views[a], views[b])
        if r is not None:
            edges.append((a, b, r))
    log(f"pairs tried {len(pairs)}, registered {len(edges)} "
        f"(E {sum(e[2]['method'] == 'E' for e in edges)}, PnP {sum(e[2]['method'] == 'PnP' for e in edges)})")
    return _layout_core(views, edges, log, first=True)


def _layout_core(views, edges, log=print, first=False):
    n = len(views)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    tree = {i: [] for i in range(n)}
    # prefer scaled edges, then more inliers
    for a, b, r in sorted(edges, key=lambda e: (-int(e[2]["scaled"]), -e[2]["inliers"])):
        if find(a) != find(b):
            parent[find(a)] = find(b)
            tree[a].append((b, r, "fwd")); tree[b].append((a, r, "rev"))
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    comps = sorted(comps.values(), key=len, reverse=True)
    log(f"components {[len(c) for c in comps][:12]}{' ...' if len(comps) > 12 else ''}")
    C, c, comp_of = {}, {}, {}
    for ci, comp in enumerate(comps):
        cal_in = [i for i in comp if views[i].cal is not None]
        if not cal_in:
            continue
        root = max(cal_in, key=lambda i: (len(tree[i]), views[i].cal["floor_share"]))
        C[root], c[root], comp_of[root] = views[root].cal["Rg"].T, np.zeros(3), ci
        # NB: cal["Rg"] maps camera->gravity frame, so camera->world = Rg (world = gravity frame)
        C[root] = views[root].cal["Rg"]
        stack = [root]
        while stack:
            j = stack.pop()
            for k, r, dirn in tree[j]:
                if k in C:
                    continue
                if dirn == "fwd":           # X_k = R X_j + t
                    Ck = C[j] @ r["R"].T
                    ck = c[j] - Ck @ r["t"]
                else:                       # X_j = R X_k + t
                    Ck = C[j] @ r["R"]
                    ck = c[j] + C[j] @ r["t"]
                v = views[k]
                if v.cal is not None:       # keep chain heading, own gravity
                    f_chain = Ck @ np.array([0, 0, 1.0]); f_own = v.cal["Rg"] @ np.array([0, 0, 1.0])
                    yaw = np.arctan2(f_chain[0], f_chain[2]) - np.arctan2(f_own[0], f_own[2])
                    Ck = Rotation.from_euler("y", yaw).as_matrix() @ v.cal["Rg"]
                elif dirn == "fwd":         # inherit depth scale from the parent through the matches
                    Xj = None
                    a_ = views[j]
                    if a_.depth is not None:
                        za = _depth_at(a_, r["pa"])
                        na = cv2.undistortPoints(r["pa"].reshape(-1, 1, 2), a_.K, None).reshape(-1, 2)
                        Xa = np.stack([na[:, 0] * za, na[:, 1] * za, za], 1)
                        zb = (Xa @ r["R"].T + r["t"])[:, 2]
                        u = np.clip((r["pb"][:, 0] / DS).astype(int), 0, v.disp.shape[1] - 1)
                        w = np.clip((r["pb"][:, 1] / DS).astype(int), 0, v.disp.shape[0] - 1)
                        x = v.disp[w, u]; y = 1.0 / np.maximum(zb, 1e-3)
                        ok = (zb > 0.2) & (za > 0.2)
                        if ok.sum() > 10:
                            sol, *_ = np.linalg.lstsq(np.stack([x[ok], np.ones(ok.sum())], 1), y[ok], rcond=None)
                            if sol[0] > 0:
                                v.depth = 1.0 / np.maximum(sol[0] * v.disp + sol[1], 1e-3)
                C[k], c[k], comp_of[k] = Ck, ck, ci
                stack.append(k)
    C, c = refine(views, edges, C, c, comp_of, log)
    if not first:
        return C, c, comp_of, comps, edges
    # hub suppression (post-mortem of the fix loop): a view whose edges mostly disagree with the global
    # solution (heading residual > 15 deg) is matching a repetitive pattern; drop its edges and re-solve
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    bad, tot = {}, {}
    for a_, b_, r in edges:
        if a_ in C and b_ in C:
            Rb_pred = C[a_] @ r["R"].T          # C_b predicted from C_a and the edge
            res = np.degrees(Rotation.from_matrix(Rb_pred @ C[b_].T).magnitude())
            for v_ in (a_, b_):
                tot[v_] = tot.get(v_, 0) + 1
                bad[v_] = bad.get(v_, 0) + (res > 15)
    hubs = {v_ for v_ in tot if tot[v_] >= 4 and bad[v_] / tot[v_] > 0.5}
    if hubs:
        log(f"hub suppression: candidate views {sorted(views[h].name for h in hubs)[:5]}")
        e2 = [(a_, b_, r) for a_, b_, r in edges if a_ not in hubs and b_ not in hubs]
        alt = layout_from_edges(views, e2, log)
        # only accept when the mapped component does not shrink by more than 20%: a hub can also be the
        # only link between rooms, and on sparse photo sets dropping it split the graph (43 -> 8 photos)
        if len(alt[3][0]) >= 0.8 * len(comps[0]):
            log("hub suppression: accepted")
            return alt
        log(f"hub suppression: rejected (main component {len(comps[0])} -> {len(alt[3][0])})")
    return C, c, comp_of, comps, edges


def layout_from_edges(views, edges, log=print):
    return _layout_core(views, edges, log)


def _yaw_of(M):
    """Heading of a rotation that is (nearly) about the vertical axis."""
    return float(np.arctan2(M[0, 2], M[2, 2]))


def refine(views, edges, C, c, comp_of, log=print, iters=30, yaw_gate=np.radians(15), t_gate=0.4):
    """Global solve after the spanning-tree initialisation.

    Each camera's tilt (gravity) is fixed - its own floor normal if it has one, else the tree's. Headings are
    averaged over ALL registered edges (Gauss-Seidel on wrapped residuals, edges off by > 15 deg ignored),
    then camera centres are solved by robust least squares over all metrically scaled edges, with every
    camera at the same height above the floor (the camera-height prior), so vertical drift cannot build up.
    """
    idx = sorted(C)
    if not idx:
        return C, c
    tilt = {i: (views[i].cal["Rg"] if views[i].cal is not None else
                Rotation.from_euler("y", -_yaw_of(C[i])).as_matrix() @ C[i]) for i in idx}
    yaw = {i: _yaw_of(C[i] @ np.linalg.inv(tilt[i])) for i in idx}
    E = [(a, b, r) for a, b, r in edges if a in C and b in C and comp_of[a] == comp_of[b]]
    meas = []
    for a, b, r in E:
        # C_b = C_a R^T  ->  Ry(yaw_b) T_b = Ry(yaw_a) T_a R^T  ->  Ry(yaw_b - yaw_a) = T_a R^T T_b^-1
        M = tilt[a] @ r["R"].T @ np.linalg.inv(tilt[b])
        meas.append((a, b, _yaw_of(M), r["inliers"]))
    roots = {}
    for i in idx:
        roots.setdefault(comp_of[i], i)
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    for _ in range(iters):
        for i in idx:
            if i in roots.values():
                continue
            sx = sy = 0.0
            for a, b, psi, w in meas:
                if b == i:
                    pred = yaw[a] + psi
                elif a == i:
                    pred = yaw[b] - psi
                else:
                    continue
                if abs(wrap(pred - yaw[i])) > yaw_gate and _ > 3:
                    continue
                sx += w * np.cos(pred); sy += w * np.sin(pred)
            if sx or sy:
                yaw[i] = float(np.arctan2(sy, sx))
    yaw_res = [abs(np.degrees(wrap(yaw[b] - yaw[a] - psi))) for a, b, psi, w in meas]
    for i in idx:
        C[i] = Rotation.from_euler("y", yaw[i]).as_matrix() @ tilt[i]
    # translations: c_b - c_a = -C_a R^T t   (scaled edges only)
    T = [(a, b, -C[a] @ r["R"].T @ r["t"], r["inliers"]) for a, b, r in E if r["scaled"] and np.linalg.norm(r["t"]) > 0]
    pos = {i: k for k, i in enumerate(idx)}
    n = len(idx)
    X = np.array([c[i][[0, 2]] for i in idx])
    w_edge = np.array([w for *_, w in T], float)
    for it in range(8):
        A = np.zeros((n, n)); B = np.zeros((n, 2))
        for (a, b, d, w0), w in zip(T, w_edge):
            ia, ib = pos[a], pos[b]
            A[ia, ia] += w; A[ib, ib] += w; A[ia, ib] -= w; A[ib, ia] -= w
            B[ib] += w * d[[0, 2]]; B[ia] -= w * d[[0, 2]]
        for r_ in roots.values():
            A[pos[r_], pos[r_]] += 1e3
            B[pos[r_]] += 1e3 * X[pos[r_]]
        A += 1e-4 * np.eye(n); B += 1e-4 * X
        X = np.linalg.solve(A, B)
        res = np.array([np.linalg.norm(X[pos[b]] - X[pos[a]] - d[[0, 2]]) for a, b, d, _ in T])
        base = np.array([w0 for *_, w0 in T], float)
        w_edge = base * np.where(res <= t_gate, 1.0, t_gate / np.maximum(res, 1e-6)) ** 2
    for i in idx:
        c[i] = np.array([X[pos[i]][0], 0.0, X[pos[i]][1]])     # equal camera height above the floor
    if T:
        log(f"refine: {len(meas)} yaw edges, median residual {np.median(yaw_res):.1f} deg; "
            f"{len(T)} translation edges, median residual {np.median(res):.2f} m, "
            f"{int((res > t_gate).sum())} down-weighted as outliers")
    return C, c


def write_pseudo(views, C, c, comp_of, out_dir, tier, extra_cols=None, prior=CAM_HEIGHT_PRIOR, fps=None):
    placed_all = [i for i in sorted(C) if views[i].depth is not None]
    main = max(set(comp_of[i] for i in placed_all), key=lambda k: sum(comp_of[i] == k for i in placed_all))
    # Only the largest connected component is mapped. Smaller components have no known position relative
    # to it; laying them out side by side (first version) produced sites hundreds of metres wide whose
    # grids exhausted memory. They are reported as unplaced instead.
    placed = [i for i in placed_all if comp_of[i] == main]
    offs = {main: np.zeros(3)}
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
    rows = []
    for fid, i in enumerate(placed):
        v = views[i]
        cv2.imwrite(os.path.join(out_dir, "depth", f"{fid:06d}.png"),
                    np.clip(v.depth * 1000, 0, 65535).astype(np.uint16))
        q = Rotation.from_matrix(C[i]).as_quat()
        p = c[i] + offs[comp_of[i]]
        s = 1920.0 / v.w
        rows.append([float(fid) / fps if fps else float(fid), fid, *p, *q, v.K[0, 0] * s, v.K[1, 1] * s,
                     v.K[0, 2] * s, v.K[1, 2] * s, v.name, v.group, comp_of[i], int(comp_of[i] == main),
                     v.cal is not None])
    df = pd.DataFrame(rows, columns=["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy",
                                     "cx", "cy", "image", "group", "component", "main_component",
                                     "self_calibrated"])
    df.to_csv(os.path.join(out_dir, "odometry.csv"), index=False)
    v0 = views[placed[0]]
    K0 = v0.K * (1920.0 / v0.w); K0[2, 2] = 1
    np.savetxt(os.path.join(out_dir, "camera_matrix.csv"), K0, delimiter=",")
    dh, dw = v0.depth.shape
    meta = {"tier": tier, "per_frame_intrinsics": True, "rgb_size": [1920, int(round(dh * 1920 / dw))],
            "depth_size": [dw, dh], "scale_prior_m": list(prior), "scale_sigma_rel": prior[1] / prior[0],
            "frames": len(placed), "main_component_frames": int(df["main_component"].sum())}
    # breaks between components (drift graph must not tie across them)
    br = df.index[df["component"].diff().fillna(0) != 0].tolist()
    meta["fragment_breaks"] = [int(x) for x in br]
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return df, meta
