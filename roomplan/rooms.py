"""Room segmentation from the carved free-space map, and rectilinear polygon extraction."""
from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage


def free_mask(free, occ, min_free=2, min_occ=2, close_m=0.25, res=0.03):
    f = (free >= min_free)
    o = occ >= min_occ
    o = cv2.dilate(o.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    m = f & ~o
    k = max(3, int(close_m / res) | 1)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    m = ndimage.binary_fill_holes(m)
    return m


def split_rooms(mask, res=0.03, min_radius=0.55, min_area=2.0):
    """Watershed on the distance transform. Seeds are maxima with clearance >= min_radius, so passages
    narrower than ~2*min_radius (doorways) separate rooms."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) * res
    seeds = (dist > min_radius).astype(np.uint8)
    # collapse each blob of high-clearance cells to one seed
    n, markers = cv2.connectedComponents(seeds)
    ws_in = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers = markers.astype(np.int32)
    markers[~mask] = -1 if False else 0
    labels = cv2.watershed(ws_in, markers.copy())
    labels[labels < 0] = 0
    labels[~mask] = 0
    out = np.zeros_like(labels)
    k = 0
    for i in range(1, labels.max() + 1):
        a = (labels == i).sum() * res * res
        if a >= min_area:
            k += 1
            out[labels == i] = k
    return out, dist


def room_polygon(room_mask, res=0.03, eps=0.10):
    m = room_mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return cv2.approxPolyDP(c, eps / res, True)[:, 0, :].astype(float)
