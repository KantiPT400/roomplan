"""Benchmark helpers that need no ground truth.

* align(planA, planB): rigid 2D alignment of two plans of the same property (4 rotations x FFT
  cross-correlation of rasterised measured walls, then sub-cell refinement on matched wall planes).
* match_rooms / match_walls: room correspondence by polygon IoU; wall correspondence by axis, position
  and overlap after alignment.
* compare(...): per-wall and per-room differences -> repeatability table (gate: |d| <= 1 cm or 0.5%).
* reference(...): the same, but against a plan treated as reference (e.g. video/photo tier vs LiDAR tier
  of the same capture). This is NOT ground-truth accuracy; reports must label it as such.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.signal import fftconvolve
from matplotlib.path import Path

ROT = np.array([[0, -1], [1, 0]])


def _segs(plan, measured_only=True):
    out = []
    for r in plan["rooms"]:
        for w in r["walls"]:
            if (not measured_only) or w["plane_measured"]:
                out.append((np.array(w["from"], float), np.array(w["to"], float)))
    return out


def _raster(segs, lo, shape, res):
    img = np.zeros(shape, np.float32)
    for a, b in segs:
        p = tuple(((a - lo) / res).astype(int)); q = tuple(((b - lo) / res).astype(int))
        cv2.line(img, p, q, 1, 2)
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def transform(plan, M, t):
    import copy
    p = copy.deepcopy(plan)
    f = lambda v: (np.asarray(v, float) @ M.T + t).round(4).tolist()
    for r in p["rooms"]:
        r["polygon_m"] = [f(v) for v in r["polygon_m"]]
        for w in r["walls"]:
            w["from"], w["to"] = f(w["from"]), f(w["to"])
    return p


def align(A, B, res=0.03):
    sa = _segs(A)
    PA = np.concatenate([np.stack(s) for s in sa])
    lo = PA.min(0) - 6; hi = PA.max(0) + 6
    shape = tuple(np.ceil((hi - lo) / res).astype(int)[::-1])
    IA = _raster(sa, lo, shape, res)
    best = None
    for rot in range(4):
        M = np.linalg.matrix_power(ROT, rot)
        sb = [(a @ M.T, b @ M.T) for a, b in _segs(B)]
        PB = np.concatenate([np.stack(s) for s in sb])
        t0 = PA.mean(0) - PB.mean(0)
        IB = _raster([(a + t0, b + t0) for a, b in sb], lo, shape, res)
        c = fftconvolve(IA, IB[::-1, ::-1], mode="same")
        i = np.unravel_index(np.argmax(c), c.shape)
        score = c[i] / np.sqrt((IA ** 2).sum() * (IB ** 2).sum())
        sh = (np.array(i) - np.array(c.shape) // 2)[::-1] * res
        if best is None or score > best[0]:
            best = (float(score), M, t0 + sh, rot)
    score, M, t, rot = best
    # refine translation with matched wall planes (median offset per axis)
    for _ in range(3):
        Bt = transform(B, M, t)
        d = {"x": [], "z": []}
        for a, b in _segs(Bt):
            ax = "x" if abs(a[0] - b[0]) < abs(a[1] - b[1]) else "z"
            for c_, e in sa:
                axa = "x" if abs(c_[0] - e[0]) < abs(c_[1] - e[1]) else "z"
                if axa != ax:
                    continue
                if ax == "x":
                    ov = min(max(a[1], b[1]), max(c_[1], e[1])) - max(min(a[1], b[1]), min(c_[1], e[1]))
                    dd = c_[0] - a[0]
                else:
                    ov = min(max(a[0], b[0]), max(c_[0], e[0])) - max(min(a[0], b[0]), min(c_[0], e[0]))
                    dd = c_[1] - a[1]
                if ov > 0.4 and abs(dd) < 0.12:
                    d[ax].append(dd)
        t = t + np.array([np.median(d["x"]) if d["x"] else 0, np.median(d["z"]) if d["z"] else 0])
    return {"score": score, "rot_deg": 90 * rot, "M": M, "t": t}


def _poly_iou(P, Q, step=0.05):
    lo = np.minimum(P.min(0), Q.min(0)); hi = np.maximum(P.max(0), Q.max(0))
    xs, zs = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step))
    g = np.stack([xs.ravel(), zs.ravel()], 1)
    a = Path(P).contains_points(g); b = Path(Q).contains_points(g)
    return (a & b).sum() / max((a | b).sum(), 1)


def match_rooms(A, B, min_iou=0.4):
    pairs = []
    for ra in A["rooms"]:
        best = None
        for rb in B["rooms"]:
            iou = _poly_iou(np.array(ra["polygon_m"]), np.array(rb["polygon_m"]))
            if best is None or iou > best[0]:
                best = (iou, rb)
        if best and best[0] >= min_iou:
            pairs.append((ra, best[1], best[0]))
    return pairs


def _wall_axis(w, poly=None):
    """(axis, coordinate, span[, facing]). facing = +1/-1: direction of the room interior along the
    wall's normal axis. Two faces of one interior wall are 10-25 cm apart and face opposite ways; they are
    different surfaces and must never be matched to each other."""
    a, b = np.array(w["from"]), np.array(w["to"])
    if abs(a[0] - b[0]) < abs(a[1] - b[1]):
        out = ("x", (a[0] + b[0]) / 2, sorted((a[1], b[1])))
    else:
        out = ("z", (a[1] + b[1]) / 2, sorted((a[0], b[0])))
    if poly is None:
        return out
    P = np.asarray(poly, float)
    cen = (a + b) / 2
    k = 0 if out[0] == "x" else 1
    probe = cen.copy(); probe[k] += 0.05
    facing = 1 if Path(P).contains_point(probe) else -1
    return out + (facing,)


def match_walls(ra, rb, pos_tol=0.15, min_ov_frac=0.5):
    out = []
    for wa in ra["walls"]:
        axa, ca, sa, fa = _wall_axis(wa, ra["polygon_m"])
        best = None
        for wb in rb["walls"]:
            axb, cb, sb, fb = _wall_axis(wb, rb["polygon_m"])
            if axa != axb or fa != fb or abs(ca - cb) > pos_tol:
                continue
            ov = min(sa[1], sb[1]) - max(sa[0], sb[0])
            if ov < min_ov_frac * min(sa[1] - sa[0], sb[1] - sb[0]):
                continue
            score = ov - abs(ca - cb)
            if best is None or score > best[0]:
                best = (score, wb)
        if best:
            out.append((wa, best[1]))
    return out


def room_dims(r):
    V = np.array(r["polygon_m"])
    e = V.max(0) - V.min(0)
    return float(e[0]), float(e[1])


def compare(A, B, label_a="A", label_b="B", only_measured=True):
    al = align(A, B)
    Bt = transform(B, al["M"], al["t"])
    rows, rooms = [], []
    for ra, rb, iou in match_rooms(A, Bt):
        rooms.append({"room_a": ra["id"], "room_b": rb["id"], "iou": round(iou, 3),
                      "area_a": ra["floor_area_m2"]["value"], "area_b": rb["floor_area_m2"]["value"],
                      "ceil_a": ra["ceiling_height_m"]["value"], "ceil_b": rb["ceiling_height_m"]["value"]})
        for wa, wb in match_walls(ra, rb):
            if only_measured and not (wa["ends_measured"] and wb["ends_measured"]):
                continue
            La, Lb = wa["length_m"], wb["length_m"]
            d = Lb["value"] - La["value"]
            rows.append({"room_a": ra["id"], "wall_a": wa["id"], "wall_b": wb["id"],
                         "len_a": La["value"], "len_b": Lb["value"], "diff_m": round(d, 4),
                         "rel": round(d / La["value"], 4),
                         "pass_repeat": bool(abs(d) <= 0.01 or abs(d) <= 0.005 * La["value"]),
                         "ci_a": La["ci95"], "ci_b": Lb["ci95"],
                         "b_in_ci_a": bool(La["ci95"][0] <= Lb["value"] <= La["ci95"][1]),
                         "a_in_ci_b": bool(Lb["ci95"][0] <= La["value"] <= Lb["ci95"][1]),
                         # z-score of the difference under both intervals: calibration check
                         "z": round(d / max(np.hypot(La["sigma"], Lb["sigma"]), 1e-6), 3)})
    return {"alignment": {"score": al["score"], "rot_deg": al["rot_deg"], "t": al["t"].round(3).tolist()},
            "rooms": rooms, "walls": rows, "aligned_b": Bt}
