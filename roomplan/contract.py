"""Assemble the published output contract (schema.json) from a tier's plan."""
from __future__ import annotations
import numpy as np


def finalize(plan, capture, tier, dbg, capture_dir, out_dir, damage=True):
    res = {"schema_version": "1.0", "capture": capture, "tier": tier, "units": "metres"}
    rooms = plan["rooms"]
    for i, r in enumerate(rooms):
        r.setdefault("name", f"Room {i + 1}")
    res["rooms"] = rooms
    res["openings"] = plan["openings"]
    res["adjacency"] = plan["adjacency"]
    res["damage"], res["concealed_damage_flags"], res["scope"] = [], [], []
    if damage:
        try:
            from .damage import detect
            res["damage"], res["concealed_damage_flags"] = detect(plan, dbg, capture_dir, tier, out_dir)
        except ImportError:
            pass
    from .scope import scope_items
    res["scope"] = scope_items(res)
    res["meta"] = plan["meta"]
    # footprint: union area of all rooms
    res["meta"]["total_floor_area_m2"] = round(sum(r["floor_area_m2"]["value"] for r in rooms), 2)
    return res
