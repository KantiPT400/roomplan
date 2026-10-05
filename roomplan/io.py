"""Loading Stray Scanner LiDAR scans (rgb.mp4, depth/*.png, odometry.csv, imu.csv, camera_matrix.csv).

Conventions (verified empirically, see docs/conventions.md):
  * odometry.csv gives camera-to-world poses; quaternion order is (qx, qy, qz, qw).
  * The camera frame is OpenCV style: x right, y down, z forward.
  * The world frame is gravity aligned with +y up (floor lies below the camera).
  * depth PNGs are uint16 millimetres at 256x192; intrinsics are for the 1920x1440 RGB image
    and must be scaled by 256/1920 for depth.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


@dataclass
class Scan:
    root: str
    poses: pd.DataFrame          # one row per frame, columns incl. frame, timestamp, x, y, z, qx..qw
    K_rgb: np.ndarray            # 3x3 intrinsics of the RGB image
    depth_size: tuple            # (w, h) of depth maps
    rgb_size: tuple              # (w, h) of the RGB image

    @property
    def K_depth(self) -> np.ndarray:
        s = self.depth_size[0] / self.rgb_size[0]
        K = self.K_rgb.copy()
        K[:2] *= s
        return K

    def K_depth_for(self, frame: int, depth_shape) -> np.ndarray:
        """Per-frame intrinsics for pseudo-LiDAR folders that set per_frame_intrinsics (odometry fx..cy are in
        units of a 1920-px-wide RGB image of the same orientation); global K otherwise."""
        if not getattr(self, "per_frame", False):
            return self.K_depth
        r = self.poses.loc[self.poses["frame"] == frame].iloc[0]
        s = depth_shape[1] / 1920.0
        return np.array([[r.fx * s, 0, r.cx * s], [0, r.fy * s, r.cy * s], [0, 0, 1.0]])

    def pose(self, frame: int):
        r = self.poses.loc[self.poses["frame"] == frame].iloc[0]
        R = Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
        return R, np.array([r.x, r.y, r.z])

    def positions(self) -> np.ndarray:
        return self.poses[["x", "y", "z"]].to_numpy()


def _read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_scan(root: str, rgb_size=(1920, 1440), depth_size=(256, 192)) -> Scan:
    meta = read_meta(root)
    if meta:
        rgb_size = tuple(meta.get("rgb_size", rgb_size))
        depth_size = tuple(meta.get("depth_size", depth_size))
    poses = _read_csv(os.path.join(root, "odometry.csv"))
    poses["frame"] = poses["frame"].astype(int)
    K = np.loadtxt(os.path.join(root, "camera_matrix.csv"), delimiter=",")
    sc = Scan(root, poses, K, depth_size, rgb_size)
    sc.per_frame = bool(meta.get("per_frame_intrinsics", False))
    return sc


def depth_frame_ids(scan: Scan, stride: int = 10) -> list[int]:
    """Frame ids that have a depth PNG on disk, subsampled by `stride` over the frame index."""
    d = os.path.join(scan.root, "depth")
    have = sorted(int(f[:-4]) for f in os.listdir(d) if f.endswith(".png"))
    known = set(scan.poses["frame"].tolist())
    return [f for f in have if f in known and f % stride == 0]


def read_depth(scan: Scan, frame: int) -> np.ndarray:
    import cv2
    p = os.path.join(scan.root, "depth", f"{frame:06d}.png")
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    return d.astype(np.float32) / 1000.0


def read_meta(root: str) -> dict:
    """meta.json is written by the video/photo tiers (pseudo-LiDAR folders); absent for real LiDAR."""
    import json
    p = os.path.join(root, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}
