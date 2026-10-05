"""Drift accountability for multi-room LiDAR captures.

ARKit VIO drifts slowly in yaw and position over a long walk, and occasionally relocalises with a
pose jump. Both show up in the fused map as doubled ("ghost") walls and walls that are a degree or two
off square. We correct this with a plane-anchored pose graph:

1. Split the trajectory into short time segments (default 6 s); always cut at pose jumps.
2. For each segment, fuse only its own depth into a wall-evidence map.
3. Measure a per-segment correction against a reference map built from the other segments:
     * yaw  : the segment's Manhattan angle minus the global one (walls are assumed orthogonal);
     * x, z : brute-force 2D cross-correlation of wall maps in a +-search window after the yaw fix;
     * y    : the segment's floor height minus the global floor.
4. Solve, per parameter, a 1-D least-squares pose graph: each segment's measurement is a soft unary
   factor weighted by its evidence; consecutive segments are tied by a smoothness factor
   (drift is slow) except across detected jumps, where the tie is dropped.
5. Interpolate the per-segment correction in time and apply it to every frame's pose.

Toggle with `--drift off` to produce the ablation ("poses used as-is").
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import fftconvolve
from .io import Scan, depth_frame_ids
from .fuse import fuse
from .planes import dominant_angle


@dataclass
class Segment:
    frames: list
    t0: float
    t1: float
    yaw: float = 0.0     # deg, measured correction
    dx: float = 0.0
    dz: float = 0.0
    dy: float = 0.0
    w_yaw: float = 0.0   # evidence weights
    w_xy: float = 0.0


def pose_jumps(scan: Scan, vmax=2.5):
    p = scan.positions()
    t = scan.poses["timestamp"].to_numpy()
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
    return scan.poses["frame"].to_numpy()[1:][v > vmax].tolist()


def split_segments(scan: Scan, frames, seg_s=6.0):
    from .io import read_meta
    jumps = set(pose_jumps(scan)) | set(read_meta(scan.root).get("fragment_breaks", []))
    ts = dict(zip(scan.poses["frame"].to_numpy(), scan.poses["timestamp"].to_numpy()))
    segs, cur, t0 = [], [], None
    allf = scan.poses["frame"].to_numpy()
    jump_after = {}
    for f in frames:
        # did a jump happen between the last frame in cur and f?
        crossed = cur and any(cur[-1] < j <= f for j in jumps)
        if cur and (ts[f] - t0 > seg_s or crossed):
            segs.append(Segment(cur, t0, ts[cur[-1]]))
            jump_after[len(segs) - 1] = bool(crossed)
            cur = []
        if not cur:
            t0 = ts[f]
        cur.append(f)
    if cur:
        segs.append(Segment(cur, t0, ts[cur[-1]]))
    return segs, jump_after


def _wall_map(P, floor_y, top_y, lo, shape, res, theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    R2 = np.array([[c, s], [-s, c]])
    sel = (P[:, 1] > floor_y + 0.3) & (P[:, 1] < top_y - 0.2)
    xz = P[sel][:, [0, 2]] @ R2.T
    ij = np.floor((xz - lo) / res).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
    m = np.zeros(shape, np.float32)
    np.add.at(m, (ij[ok, 1], ij[ok, 0]), 1.0)
    return np.log1p(m)


def _xcorr_shift(ref, mov, max_px):
    """Shift (dx, dz) in pixels that best aligns mov onto ref, searched in +-max_px; returns shift, peak ratio."""
    c = fftconvolve(ref, mov[::-1, ::-1], mode="same")
    cy, cx = np.array(c.shape) // 2
    win = c[cy - max_px:cy + max_px + 1, cx - max_px:cx + max_px + 1]
    i = np.unravel_index(np.argmax(win), win.shape)
    peak = win[i]
    # quadratic sub-pixel refinement
    def sub(a, b, cc):
        d = a - 2 * cc + b
        return 0.0 if d == 0 else 0.5 * (a - b) / d
    dy = i[0] - max_px + (sub(win[i[0] - 1, i[1]], win[i[0] + 1, i[1]], peak) if 0 < i[0] < win.shape[0] - 1 else 0)
    dx = i[1] - max_px + (sub(win[i[0], i[1] - 1], win[i[0], i[1] + 1], peak) if 0 < i[1] < win.shape[1] - 1 else 0)
    med = np.median(win) + 1e-6
    return (dx, dy), float(peak / med)


def _smooth(meas, w, tie, breaks):
    """argmin sum w_i (x_i - m_i)^2 + tie * sum (x_i - x_{i+1})^2 (no tie across breaks)."""
    n = len(meas)
    A = np.diag(np.asarray(w, float) + 1e-6)
    for i in range(n - 1):
        if breaks.get(i):
            continue
        A[i, i] += tie; A[i + 1, i + 1] += tie
        A[i, i + 1] -= tie; A[i + 1, i] -= tie
    return np.linalg.solve(A, np.asarray(w, float) * np.asarray(meas, float))


def wall_facing(walls, Q, cam_xz, tol=0.06):
    """For each wall, which side the observing cameras were on (+1 / -1) along its normal axis.
    Q: rotated points, cam_xz: per-point camera position in the same rotated xz frame."""
    out = []
    for w in walls:
        if w.axis == "x":
            sel = (np.abs(Q[:, 0] - w.coord) < tol) & (Q[:, 2] > w.lo) & (Q[:, 2] < w.hi)
            side = np.sign(np.median(cam_xz[sel, 0] - w.coord)) if sel.any() else 0
        else:
            sel = (np.abs(Q[:, 2] - w.coord) < tol) & (Q[:, 0] > w.lo) & (Q[:, 0] < w.hi)
            side = np.sign(np.median(cam_xz[sel, 1] - w.coord)) if sel.any() else 0
        out.append(int(side))
    return out


def _seg_walls(scan, frames, floor_y, top_y, theta, corr=None):
    from .walls import extract_walls
    from .planes import rotate_xz
    P, F = fuse(scan, frames=frames, voxel=0.015)
    if corr is not None:
        P = apply_to_points(P, *corr, theta=theta, pivot=_pivot_frames(scan, frames))
    Q, R2 = rotate_xz(P, theta)
    if len(Q) < 500:
        return [], []
    walls, *_ = extract_walls(Q, floor_y, top_y, min_len=0.3)
    pos = dict(zip(scan.poses["frame"].to_numpy(), scan.positions()))
    cam = np.array([pos[f] for f in F])
    if corr is not None:
        cam = apply_to_points(cam, *corr, theta=theta, pivot=_pivot_frames(scan, frames))
    cam_xz = cam[:, [0, 2]] @ R2.T
    return walls, wall_facing(walls, Q, cam_xz)


def _match_shift(seg_walls, seg_face, ref_walls, ref_face, axis, tol=0.15, min_ov=0.3):
    d, w = [], []
    for a, fa in zip(seg_walls, seg_face):
        if a.axis != axis or fa == 0:
            continue
        best = None
        for b, fb in zip(ref_walls, ref_face):
            if b.axis != axis or fb != fa:
                continue
            ov = min(a.hi, b.hi) - max(a.lo, b.lo)
            if ov < min_ov:
                continue
            dd = a.coord - b.coord
            if abs(dd) < tol and (best is None or abs(dd) < abs(best[0])):
                best = (dd, ov)
        if best:
            d.append(best[0]); w.append(best[1])
    if not d:
        return 0.0, 0.0
    d, w = np.array(d), np.array(w)
    o = np.argsort(d); c = np.cumsum(w[o]) / w.sum()
    med = d[o][np.searchsorted(c, 0.5)]
    return float(med), float(min(w.sum(), 4.0) / 4.0)


def estimate(scan: Scan, floor_y, top_y, theta, stride=10, seg_s=6.0, frames=None, tie=2.0,
             tol=0.20, huber=0.03, min_gap_s=0.0):
    """Per-segment (yaw, dx, dz, dy) corrections from a pose graph over time segments.

    * yaw, dy : absolute unary factors (walls are orthogonal, the floor is level), smoothed in time.
    * dx, dz  : relative factors between EVERY pair of segments that see a common wall face
                (same axis, same observed side, overlapping >= 0.3 m, within `tol`). Pairs far apart in
                time are exactly the loop closures. Solved jointly by robust (Huber-IRLS) least squares,
                with a weak smoothness tie between consecutive segments (dropped across pose jumps) and the
                mean correction fixed to zero (gauge).
    """
    frames = frames if frames is not None else depth_frame_ids(scan, stride)
    segs, breaks = split_segments(scan, frames, seg_s)
    n = len(segs)
    for sg in segs:
        P = fuse(scan, frames=sg.frames, voxel=0.02)[0]
        sl = (P[:, 1] > floor_y + 0.3) & (P[:, 1] < top_y - 0.2)
        if sl.sum() > 400:
            th = dominant_angle(P, floor_y + 0.3, top_y - 0.2)
            sg.yaw = float(((th - theta + 45) % 90) - 45)
            sg.w_yaw = float(min(sl.sum(), 4000)) / 4000.0
        fl = P[np.abs(P[:, 1] - floor_y) < 0.08, 1]
        sg.dy = float(np.median(fl) - floor_y) if len(fl) > 200 else 0.0
    # Heading is NOT corrected. Per-segment "Manhattan" yaw mixes drift with real non-squareness of walls
    # (a wall 0.5 deg off square shifts the estimate of every segment that mostly sees it); applying it bent
    # the map: loop yaw residual 0.04 deg uncorrected -> 0.45 deg corrected on with_ceiling. The measured
    # per-segment yaw is kept for reporting only.
    yaw_meas = _smooth([s.yaw for s in segs], [s.w_yaw for s in segs], tie * 40, breaks)
    yaw_s = np.zeros(n)
    dy_s = _smooth([s.dy for s in segs], [1.0] * n, tie * 2, breaks)
    sw = [_seg_walls(scan, sg.frames, floor_y, top_y, theta, (yaw_s[k], 0, 0, dy_s[k]))
          for k, sg in enumerate(segs)]
    edges = {"x": [], "z": []}
    for i in range(n):
        for j in range(i + 1, n):
            if min_gap_s and segs[j].t0 - segs[i].t1 < min_gap_s and j != i + 1:
                continue
            for ax in ("x", "z"):
                d, w = _match_shift(sw[j][0], sw[j][1], sw[i][0], sw[i][1], ax, tol=tol)
                if w > 0.05:
                    edges[ax].append((i, j, d, w))
    sol = {}
    for ax in ("x", "z"):
        E = edges[ax]
        t = np.zeros(n)
        for _ in range(6):
            A = np.zeros((n, n)); b = np.zeros(n)
            for i, j, d, w in E:
                r = t[j] - t[i] - d
                wr = w * (1.0 if abs(r) <= huber else huber / abs(r))
                A[i, i] += wr; A[j, j] += wr; A[i, j] -= wr; A[j, i] -= wr
                b[i] -= wr * d; b[j] += wr * d
            for k in range(n - 1):
                if not breaks.get(k):
                    A[k, k] += tie * 0.05; A[k + 1, k + 1] += tie * 0.05
                    A[k, k + 1] -= tie * 0.05; A[k + 1, k] -= tie * 0.05
            A += 1.0 / n          # gauge: penalise the mean (sum t)^2
            A += 1e-3 * np.eye(n)  # weak zero prior: keeps components with no edges solvable
            t = np.linalg.solve(A, b)
        sol[ax] = t
        sol[ax + "_edges"] = len(E)
        sol[ax + "_resid"] = float(np.sqrt(np.mean([(t[j] - t[i] - d) ** 2 for i, j, d, w in E]))) if E else 0.0
    for k, sg in enumerate(segs):
        sg.yaw, sg.dx, sg.dz, sg.dy = float(yaw_s[k]), float(sol["x"][k]), float(sol["z"][k]), float(dy_s[k])
        sg.yaw_measured = float(yaw_meas[k])
    info = {"yaw_measured_range_deg": float(np.ptp(yaw_meas)), "yaw_corrected": False, "n_segments": n, "jumps": int(sum(bool(v) for v in breaks.values())),
            "edges_x": sol["x_edges"], "edges_z": sol["z_edges"],
            "rms_edge_residual_x_m": sol["x_resid"], "rms_edge_residual_z_m": sol["z_resid"]}
    return segs, breaks, info


def _pivot_frames(scan: Scan, frames):
    p = scan.positions()[np.isin(scan.poses["frame"].to_numpy(), frames)]
    return p.mean(0)


def _pivot(scan: Scan, sg: Segment):
    return _pivot_frames(scan, sg.frames)


def apply_to_points(P, yaw_deg, dx, dz, dy, theta, pivot):
    """Undo a measured segment error (yaw_deg, dx, dz, dy): yaw_deg is how far the segment's Manhattan angle
    sits above the global one, (dx, dz) how far its walls sit from the global walls in the Manhattan frame,
    dy its floor offset. Verified: applying yaw_deg=+1 lowers the measured Manhattan angle by ~1 deg."""
    Rw = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
    Q = (P - pivot) @ Rw.T + pivot
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    R2 = np.array([[c, s], [-s, c]])
    shift = np.array([dx, dz]) @ R2      # Manhattan-frame shift back to world xz
    Q[:, 0] -= shift[0]; Q[:, 2] -= shift[1]
    Q[:, 1] -= dy
    return Q


def corrected_poses(scan: Scan, segs, theta):
    """Return a copy of scan.poses with the per-segment correction interpolated in time and applied."""
    df = scan.poses.copy()
    t = df["timestamp"].to_numpy()
    tc = np.array([(s.t0 + s.t1) / 2 for s in segs])
    piv = np.array([_pivot(scan, s) for s in segs])
    def interp(v):
        return np.interp(t, tc, v)
    yaw = interp([s.yaw for s in segs]); dx = interp([s.dx for s in segs]); dz = interp([s.dz for s in segs])
    dy = interp([s.dy for s in segs])
    px, py, pz = interp(piv[:, 0]), interp(piv[:, 1]), interp(piv[:, 2])
    P = df[["x", "y", "z"]].to_numpy()
    q = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].to_numpy())
    newP = np.empty_like(P)
    newq = np.empty((len(df), 4))
    c, s_ = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    R2 = np.array([[c, s_], [-s_, c]])
    for i in range(len(df)):
        Rw = Rotation.from_euler("y", yaw[i], degrees=True)
        piv_i = np.array([px[i], py[i], pz[i]])
        p = Rw.apply(P[i] - piv_i) + piv_i
        sh = np.array([dx[i], dz[i]]) @ R2
        p[0] -= sh[0]; p[2] -= sh[1]; p[1] -= dy[i]
        newP[i] = p
        newq[i] = (Rw * q[i]).as_quat()
    df[["x", "y", "z"]] = newP
    df[["qx", "qy", "qz", "qw"]] = newq
    return df


def wall_sharpness(walls):
    """Length-weighted median of wall-face spread (m). Ghost walls from drift inflate this."""
    if not walls:
        return float("nan")
    s = np.array([w.std for w in walls]); L = np.array([w.length for w in walls])
    o = np.argsort(s); c = np.cumsum(L[o]) / L.sum()
    return float(s[o][np.searchsorted(c, 0.5)])


def ghost_walls(walls, lo=0.04, hi=0.09, min_ov=0.4):
    """Pairs of parallel wall planes 4-9 cm apart overlapping >= 0.4 m. Real interior walls are thicker
    than ~9 cm, so planes this close are almost always one surface seen twice under different drift."""
    n = 0
    for i, a in enumerate(walls):
        for b in walls[i + 1:]:
            if a.axis == b.axis and lo < abs(a.coord - b.coord) < hi:
                if min(a.hi, b.hi) - max(a.lo, b.lo) >= min_ov:
                    n += 1
    return n
