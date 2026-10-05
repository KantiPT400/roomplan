"""Metric, gravity-aligned geometry from ONE photo (no SfM needed).

Mono depth gives relative inverse depth d: true 1/z = a*(d_n + beta) with d_n = d / p99(d).
  * beta (shape) : chosen so that the back-projected scene is most planar and its planes most nearly
                   parallel/orthogonal (indoor scenes are Manhattan). Checked against LiDAR on 7 frames:
                   median depth error 3-6% with the recovered beta vs 19-26% with beta = 0.
  * gravity      : normal of the floor plane (the large plane below the camera whose normal is closest to
                   image-up; photos are upright via EXIF orientation).
  * a (scale)    : camera height above the floor = CAM_HEIGHT_PRIOR (pseudo.py).
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .pseudo import _ransac_planes, CAM_HEIGHT_PRIOR


def _cloud(dn, K, beta, step=4):
    z = 1.0 / np.maximum(dn + beta, 1e-3)
    h, w = z.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    zz = z[::step, ::step]
    return np.stack([(u - K[0, 2]) / K[0, 0] * zz, (v - K[1, 2]) / K[1, 1] * zz, zz], -1).reshape(-1, 3)


def _shape_score(P, rng):
    P = P / np.median(P[:, 2])
    pl = _ransac_planes(P, thr=0.02, n_planes=3, iters=150, min_frac=0.05, rng=rng)
    inl = sum(c for _, _, c in pl) / len(P)
    orth = 0.0
    for i in range(len(pl)):
        for j in range(i + 1, len(pl)):
            c = abs(pl[i][0] @ pl[j][0])
            orth += min(c, 1 - c)
    return inl - 0.5 * orth


def calibrate(d, K, prior=CAM_HEIGHT_PRIOR, seed=0):
    """d: disparity at the resolution K refers to. Returns dict or None."""
    rng = np.random.default_rng(seed)
    p99 = np.percentile(d, 99)
    dn = d / max(p99, 1e-6)
    lo = max(-0.9 * float(np.percentile(dn, 1)), -0.3)
    betas = np.linspace(lo, 1.6, 18)
    sc = [_shape_score(_cloud(dn, K, b, 6), rng) for b in betas]
    b0 = betas[int(np.argmax(sc))]
    fine = np.linspace(b0 - 0.1, b0 + 0.1, 9)
    fine = fine[fine > lo]
    sc2 = [_shape_score(_cloud(dn, K, b, 4), rng) for b in fine]
    beta = float(fine[int(np.argmax(sc2))])
    P = _cloud(dn, K, beta, 4)
    P = P / np.median(P[:, 2])
    planes = _ransac_planes(P, thr=0.015, n_planes=5, iters=300, min_frac=0.03, rng=rng)
    up_img = np.array([0, -1.0, 0])
    best = None
    for nrm, dd, cnt in planes:
        # orient normal towards the camera (origin): camera is on the side where n.x + d > 0
        if dd < 0:
            nrm, dd = -nrm, -dd
        if nrm @ up_img < np.cos(np.radians(65)):      # floor normal must point roughly image-up
            continue
        # the floor is the LOWEST large near-horizontal plane: table, bed and counter tops are closer
        if cnt / len(P) < 0.08:       # less floor than this is not a reliable self-calibration
            continue
        score = dd
        if best is None or score > best[0]:
            best = (score, nrm, dd, cnt)
    if best is None:
        return None
    _, nrm, dd, cnt = best
    h_units = dd                                       # distance camera -> floor plane, in relative units
    s = prior[0] / h_units                             # metres per relative unit
    Rg, _ = Rotation.align_vectors([[0, 1, 0]], [nrm])  # camera coords -> gravity-aligned (y up)
    tilt = float(np.degrees(np.arccos(np.clip(nrm @ up_img, -1, 1))))
    return {"beta": beta, "p99": float(p99), "scale": float(s / np.median(_cloud(dn, K, beta, 4)[:, 2])),
            "Rg": Rg.as_matrix(), "floor_share": float(cnt / len(P)), "tilt_deg": tilt}


def depth_from(d, cal):
    """Metric depth map (metres) from disparity using a calibration."""
    dn = d / cal["p99"]
    return cal["scale"] / np.maximum(dn + cal["beta"], 1e-3)
