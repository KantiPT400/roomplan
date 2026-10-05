"""Monocular relative depth (Depth Anything V2 small, ONNX, CPU).

Weights are fetched by scripts/fetch_models.sh from the public GitHub release of
fabio-sim/Depth-Anything-ONNX (v2.0.0, depth_anything_v2_vits.onnx, sha256 recorded there).
Output is relative inverse depth ("disparity"): metric depth z satisfies 1/z = a*disp + b for an
unknown per-image (a, b), which we recover from SfM points (see pseudo.py).
Results are cached per image as .npy so re-runs replay deterministically.
"""
from __future__ import annotations
import os
import hashlib
import numpy as np
import cv2

MODEL = os.environ.get("ROOMPLAN_DEPTH_MODEL",
                       os.path.join(os.path.dirname(__file__), "..", "models", "depth_anything_v2_vits.onnx"))
_SESS = None


def _session():
    global _SESS
    if _SESS is None:
        import onnxruntime as ort
        if not os.path.exists(MODEL):
            raise FileNotFoundError(f"{MODEL} missing: run scripts/fetch_models.sh")
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 2
        _SESS = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    return _SESS


def disparity(img_bgr: np.ndarray, size=518) -> np.ndarray:
    x = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    s = _session()
    d = s.run(None, {s.get_inputs()[0].name: x})[0][0]
    return cv2.resize(d, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def disparity_cached(path: str, cache_dir: str, out_size=(256, 192)) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1((os.path.abspath(path) + str(os.path.getmtime(path))).encode()).hexdigest()[:16]
    c = os.path.join(cache_dir, key + ".npy")
    if os.path.exists(c):
        return np.load(c)
    img = cv2.imread(path)
    d = disparity(img)
    d = cv2.resize(d, out_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    np.save(c, d)
    return d
