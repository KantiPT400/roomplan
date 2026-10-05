"""Loading Record3D-style LiDAR scans (rgb.mp4, depth/*.png, odometry.csv, imu.csv, camera_matrix.csv).

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
    poses = _read_csv(os.path.join(root, "odometry.csv"))
    poses["frame"] = poses["frame"].astype(int)
    K = np.loadtxt(os.path.join(root, "camera_matrix.csv"), delimiter=",")
    return Scan(root, poses, K, depth_size, rgb_size)


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
