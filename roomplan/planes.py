"""Gravity-aligned structure: floor, ceiling, wall directions, footprint polygon."""
from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage


def _peak(h, edges, lo, hi, min_frac, n):
    """Highest histogram bin centre in [lo, hi] whose mass exceeds min_frac of n."""
    c = 0.5 * (edges[:-1] + edges[1:])
    m = (c >= lo) & (c <= hi)
    if not m.any():
        return None
    # smooth 3 bins so a plane split across two bins still counts
    hs = np.convolve(h, np.ones(3), "same")
    i = np.argmax(np.where(m, hs, -1))
    return (c[i], hs[i] / n) if hs[i] / n >= min_frac else None


def find_floor_ceiling(P, cam_y, bin_=0.01, floor_frac=0.04, ceil_frac=0.03):
    """Floor/ceiling heights (world y) from horizontal-plane peaks.

    Floor is the strongest peak 0.8-2.2 m below the camera; ceiling the strongest 0.5-2.0 m above.
    Each is refined as the median of points within 3 cm. Returns dict with y and inlier counts.
    """
    edges = np.arange(P[:, 1].min() - 0.1, P[:, 1].max() + 0.1, bin_)
    h, _ = np.histogram(P[:, 1], edges)
    out = {"floor": None, "ceiling": None}
    for name, (lo, hi), frac in (("floor", (cam_y - 2.2, cam_y - 0.8), floor_frac),
                                 ("ceiling", (cam_y + 0.5, cam_y + 2.0), ceil_frac)):
        pk = _peak(h, edges, lo, hi, frac, len(P))
        if pk is None:
            continue
        y0, share = pk
        sel = np.abs(P[:, 1] - y0) < max(0.03, 1.5 * bin_)
        out[name] = {"y": float(np.median(P[sel, 1])), "n": int(sel.sum()), "share": float(share),
                     "std": float(P[sel, 1].std())}
    return out


def dominant_angle(P, y_lo, y_hi, grid=0.03):
    """Manhattan angle theta in [0, 90) deg from gradient orientation of the wall-slab occupancy."""
    S = P[(P[:, 1] > y_lo) & (P[:, 1] < y_hi)][:, [0, 2]]
    mn = S.min(0)
    ij = np.floor((S - mn) / grid).astype(int)
    img = np.zeros(ij.max(0)[::-1] + 1, np.float32)
    np.add.at(img, (ij[:, 1], ij[:, 0]), 1)
    img = cv2.GaussianBlur(np.log1p(img), (0, 0), 1.5)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 90.0
    # circular mean on the 90-degree period
    w = mag.ravel()
    a = np.radians(ang.ravel() * 4.0)
    th = np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) / 4.0
    return th % 90.0


def rotate_xz(P, theta_deg):
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    R = np.array([[c, s], [-s, c]])
    Q = P.copy()
    Q[:, [0, 2]] = P[:, [0, 2]] @ R.T
    return Q, R


def footprint(P_rot, floor_y, cam_xz, grid=0.02, tol=0.04, close_m=0.5):
    """Rectilinear footprint from floor points plus the camera path.

    Returns (polygon Mx2 in rotated xz metres, mask, origin). Floor points are the observed
    floor; closing bridges furniture occlusions; the camera path seeds the room interior.
    """
    F = P_rot[np.abs(P_rot[:, 1] - floor_y) < tol][:, [0, 2]]
    mn = F.min(0) - 0.5
    sz = np.ceil((F.max(0) + 0.5 - mn) / grid).astype(int)
    img = np.zeros(sz[::-1], np.uint8)
    ij = np.floor((F - mn) / grid).astype(int)
    img[ij[:, 1], ij[:, 0]] = 255
    k = int(close_m / grid)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    img = ndimage.binary_fill_holes(img > 0).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(img)
    cij = np.floor((cam_xz - mn) / grid).astype(int)
    cij = cij[(cij[:, 0] >= 0) & (cij[:, 1] >= 0) & (cij[:, 0] < sz[0]) & (cij[:, 1] < sz[1])]
    keep = set(lab[cij[:, 1], cij[:, 0]].tolist()) - {0}
    if not keep:
        keep = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}
    mask = np.isin(lab, list(keep)).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 0.08 / grid, True)[:, 0, :] * grid + mn
    return poly, mask, mn
