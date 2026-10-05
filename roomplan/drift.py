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
    jumps = set(pose_jumps(scan))
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


def estimate(scan: Scan, floor_y, top_y, theta, stride=10, seg_s=6.0, res=0.02, search=0.35,
             frames=None, iters=2, tie=4.0):
    frames = frames if frames is not None else depth_frame_ids(scan, stride)
    segs, breaks = split_segments(scan, frames, seg_s)
    clouds = [fuse(scan, frames=s.frames, voxel=0.02)[0] for s in segs]
    allP = np.concatenate(clouds)
    c, s_ = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    R2 = np.array([[c, s_], [-s_, c]])
    xz = allP[:, [0, 2]] @ R2.T
    lo = xz.min(0) - 1.0
    shape = tuple(np.ceil((xz.max(0) + 1.0 - lo) / res).astype(int)[::-1])
    # 1) yaw and height per segment
    for sg, P in zip(segs, clouds):
        sl = (P[:, 1] > floor_y + 0.3) & (P[:, 1] < top_y - 0.2)
        sg.w_yaw = float(min(sl.sum(), 4000)) / 4000.0
        if sl.sum() > 400:
            th = dominant_angle(P, floor_y + 0.3, top_y - 0.2)
            sg.yaw = float(((th - theta + 45) % 90) - 45)
        else:
            sg.w_yaw = 0.0
        fl = P[np.abs(P[:, 1] - floor_y) < 0.08, 1]
        sg.dy = float(np.median(fl) - floor_y) if len(fl) > 200 else 0.0
    yaw_s = _smooth([s.yaw for s in segs], [s.w_yaw for s in segs], tie, breaks)
    dy_s = _smooth([s.dy for s in segs], [1.0] * len(segs), tie, breaks)
    # 2) translation per segment, iterated against a leave-one-out reference
    tx = np.zeros(len(segs)); tz = np.zeros(len(segs))
    for it in range(iters):
        maps = []
        for k, (sg, P) in enumerate(zip(segs, clouds)):
            Pc = apply_to_points(P, -yaw_s[k], tx[k], tz[k], dy_s[k], theta, pivot=_pivot(scan, sg))
            maps.append(_wall_map(Pc, floor_y, top_y, lo, shape, res, theta))
        total = np.sum(maps, axis=0)
        mx, mz, wx = np.zeros(len(segs)), np.zeros(len(segs)), np.zeros(len(segs))
        for k, m in enumerate(maps):
            if m.sum() < 50:
                continue
            ref = total - m
            (dx, dz), ratio = _xcorr_shift(ref, m, int(search / res))
            mx[k], mz[k] = dx * res, dz * res
            wx[k] = float(np.clip((ratio - 1.0) / 4.0, 0, 1))
        tx = tx + _smooth(mx, wx, tie, breaks)
        tz = tz + _smooth(mz, wx, tie, breaks)
    for k, sg in enumerate(segs):
        sg.yaw, sg.dx, sg.dz, sg.dy = float(yaw_s[k]), float(tx[k]), float(tz[k]), float(dy_s[k])
    return segs, breaks


def _pivot(scan: Scan, sg: Segment):
    p = scan.positions()[np.isin(scan.poses["frame"].to_numpy(), sg.frames)]
    return p.mean(0)


def apply_to_points(P, yaw_deg, dx, dz, dy, theta, pivot):
    """Rotate about the vertical axis through pivot by yaw_deg, then shift (dx, dz) in the Manhattan frame
    and dy vertically. Returns new points in the original world frame."""
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
    yaw = interp([-s.yaw for s in segs]); dx = interp([s.dx for s in segs]); dz = interp([s.dz for s in segs])
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
