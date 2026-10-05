"""Scope line items keyed to surfaces. Quantities carry the interval of the measurement they derive from."""
from __future__ import annotations
import numpy as np
from .uncertainty import ci


def scope_items(res):
    items = []
    for r in res["rooms"]:
        H = r["ceiling_height_m"]
        h = H["value"] if H["value"] is not None else 2.7
        hs = H["sigma"] if H.get("sigma") else 0.25
        A = r["floor_area_m2"]
        items.append({"surface": f"{r['id']}/floor", "item": "floor area (measure for finish)",
                      "quantity": A, "unit": "m2", "reason": "base quantity"})
        items.append({"surface": f"{r['id']}/ceiling", "item": "ceiling area (paint)",
                      "quantity": A, "unit": "m2", "reason": "base quantity"})
        for w in r["walls"]:
            L, Ls = w["length_m"]["value"], w["length_m"]["sigma"]
            a = L * h
            s = float(np.sqrt((Ls * h) ** 2 + (L * hs) ** 2))
            items.append({"surface": w["id"], "item": "wall area (paint, gross of openings)",
                          "quantity": ci(a, s, 2), "unit": "m2", "reason": "base quantity"})
    for d in res.get("damage", []):
        e = d["extent_m2"]
        if d["class"] in ("stain_discoloration", "mould_like"):
            items.append({"surface": d["surface"], "item": "stain-block primer + repaint, damaged patch +0.3 m margin",
                          "quantity": ci(e["value"] + 0.3 * 4 * np.sqrt(max(e["value"], 1e-4)) + 0.36,
                                         e["sigma"] or 0.05, 2), "unit": "m2", "reason": d["id"]})
        elif d["class"] == "crack":
            items.append({"surface": d["surface"], "item": "rake out, fill and repaint crack",
                          "quantity": d.get("length_m", ci(0.0, 0.0)), "unit": "m", "reason": d["id"]})
        else:
            items.append({"surface": d["surface"], "item": "patch plaster and repaint",
                          "quantity": e, "unit": "m2", "reason": d["id"]})
    for f in res.get("concealed_damage_flags", []):
        items.append({"surface": f["surface"], "item": "moisture meter check before closing up (inspection)",
                      "quantity": ci(1, 0, 0), "unit": "ea", "reason": f["rule"]})
    return items
