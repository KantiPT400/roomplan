"""Render a dimensioned floor plan (PNG/SVG)."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#dbe8f4", "#f4e3d0", "#dcefdc", "#efe0f0", "#f6f0c8", "#d8eeee", "#f2d9d9", "#e6e6e6"]


def _label_point(V, res=0.02):
    """Interior point farthest from the room outline (an L-shaped room's vertex mean can lie outside it),
    and that distance (how much room there is for text)."""
    import cv2
    lo = V.min(0) - 2 * res
    P = np.round((V - lo) / res).astype(np.int32)
    img = np.zeros((P[:, 1].max() + 3, P[:, 0].max() + 3), np.uint8)
    cv2.fillPoly(img, [P], 1)
    d = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    i = np.unravel_index(np.argmax(d), d.shape)
    return lo + np.array([i[1], i[0]]) * res, float(d[i]) * res


def render(plan, path, title=None, points=None):
    allv = np.concatenate([np.array(r["polygon_m"]) for r in plan["rooms"]]) if plan["rooms"] else np.zeros((1, 2))
    span = np.ptp(allv, 0) + 2.0
    s_ = 11.0 / max(span.max(), 1e-6)
    fig, ax = plt.subplots(figsize=(max(5.0, span[0] * s_), max(5.0, span[1] * s_ + 0.6)))
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], s=0.05, c="#bbbbbb", zorder=0)
    for i, r in enumerate(plan["rooms"]):
        V = np.array(r["polygon_m"])
        ax.fill(V[:, 0], V[:, 1], color=PALETTE[i % len(PALETTE)], zorder=1)
        c, room_r = _label_point(V)
        small = room_r < 0.8                       # narrow room: name + area only, short walls unlabelled
        for w in r["walls"]:
            a, b = np.array(w["from"]), np.array(w["to"])
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#222" if w["ends_measured"] else "#999",
                    lw=2.5 if w["plane_measured"] else 1.2, zorder=3)
            L = w["length_m"]
            if L["value"] >= (0.8 if small else 0.4):
                m = (a + b) / 2
                d = b - a
                nrm = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6)
                # put the label inside the room
                if not _inside(V, m + 0.05 * nrm):
                    nrm = -nrm
                p = m + (0.14 if small else 0.22) * nrm
                rot = 0 if abs(d[0]) > abs(d[1]) else 90
                half = (L["ci95"][1] - L["ci95"][0]) / 2
                ax.text(p[0], p[1], f"{L['value']:.2f}±{half:.2f}", ha="center", va="center",
                        rotation=rot, fontsize=6 if small else 7, zorder=5)
        A = r["floor_area_m2"]
        H = r["ceiling_height_m"]
        htxt = f"h {H['value']:.2f} m" if H["value"] is not None else "h n/a"
        name = r.get("name", r["id"])
        if r.get("status") == "single_view_estimate":
            name += " (estimate)"
        txt = f"{name}\n{A['value']:.1f} m²" + ("" if small else f"\n{htxt}")
        ax.text(c[0], c[1], txt, ha="center", va="center", fontsize=7 if small else 9, weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6))
    for o in plan.get("openings", []):
        lo, hi = o["span"]
        if o["axis"] == "x":
            xs, zs = [o["coord"]] * 2, [lo, hi]
        else:
            xs, zs = [lo, hi], [o["coord"]] * 2
        ax.plot(xs, zs, "-", color="#c0392b", lw=5, solid_capstyle="butt", zorder=4)
        ax.text(np.mean(xs), np.mean(zs), f"{o['width_m']['value']:.2f}", color="#c0392b", fontsize=7,
                ha="center", va="center", zorder=7, weight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="#c0392b", lw=0.5, alpha=0.85))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("m"); ax.set_ylabel("m")
    ax.set_title(title or "Floor plan (95% intervals)")
    fig.text(0.01, 0.005, "thick wall: plane measured, thin: inferred;  dark: both ends measured, grey: an end bounded "
             "by furniture or unseen (wider interval)\nred: opening width (m);  lengths: value ± 95% half-interval "
             "(m);  narrow rooms: short walls unlabelled;  all numbers in plan.json", fontsize=6.5, color="#444")
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _inside(V, p):
    from matplotlib.path import Path
    return Path(V).contains_point(p)
