"""Structure from motion with pycolmap. Returns one or more fragments (indoor video of white walls
routinely breaks into several models; we keep all of them and merge later)."""
from __future__ import annotations
import os
import shutil
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Fragment:
    names: list                 # image names, time ordered
    R: dict                     # name -> 3x3 world-to-camera rotation
    t: dict                     # name -> 3 world-to-camera translation
    K: np.ndarray               # 3x3 intrinsics for the SfM image size
    size: tuple                 # (w, h) of the SfM images
    obs: dict                   # name -> (uv Nx2, X Nx3) observations of triangulated points
    points: np.ndarray = None   # Mx3

    def center(self, n):
        return -self.R[n].T @ self.t[n]


def run_sfm(image_dir, work, focal_px, sequential=True, overlap=25, min_model=6, quiet=True):
    import pycolmap
    os.makedirs(work, exist_ok=True)
    db = os.path.join(work, "db.db")
    if os.path.exists(db):
        os.remove(db)
    names = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic")))
    import cv2
    h, w = cv2.imread(os.path.join(image_dir, names[0])).shape[:2]
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "SIMPLE_RADIAL"
    ro.camera_params = f"{focal_px},{w / 2},{h / 2},0"
    eo = pycolmap.FeatureExtractionOptions()
    eo.sift.max_num_features = 4096
    eo.num_threads = os.cpu_count() or 2
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro,
                              extraction_options=eo)
    mo = pycolmap.FeatureMatchingOptions()
    mo.guided_matching = True
    if sequential:
        so = pycolmap.SequentialPairingOptions()
        so.overlap = overlap
        so.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=so, matching_options=mo)
    else:
        pycolmap.match_exhaustive(db, matching_options=mo)
    out = os.path.join(work, "sparse")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = min_model
    opts.mapper.init_min_num_inliers = 60
    recs = pycolmap.incremental_mapping(db, image_dir, out, options=opts)
    frags = []
    for _, rec in recs.items():
        cam = next(iter(rec.cameras.values()))
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        Rd, td, obs = {}, {}, {}
        for img in rec.images.values():
            if not img.has_pose:
                continue
            T = img.cam_from_world()
            M = T.matrix()
            Rd[img.name], td[img.name] = M[:, :3].copy(), M[:, 3].copy()
            uv, X = [], []
            for p2 in img.points2D:
                if p2.has_point3D():
                    uv.append(p2.xy); X.append(rec.points3D[p2.point3D_id].xyz)
            obs[img.name] = (np.array(uv).reshape(-1, 2), np.array(X).reshape(-1, 3))
        pts = np.array([p.xyz for p in rec.points3D.values()])
        frags.append(Fragment(sorted(Rd), Rd, td, K, (w, h), obs, pts))
    frags.sort(key=lambda fr: fr.names[0])
    return frags
