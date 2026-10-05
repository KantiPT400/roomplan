"""Render a dimensioned floor plan (PNG/SVG)."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#dbe8f4", "#f4e3d0", "#dcefdc", "#efe0f0", "#f6f0c8", "#d8eeee", "#f2d9d9", "#e6e6e6"]


def render(plan, path, title=None, points=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], s=0.05, c="#bbbbbb", zorder=0)
    for i, r in enumerate(plan["rooms"]):
        V = np.array(r["polygon_m"])
        ax.fill(V[:, 0], V[:, 1], color=PALETTE[i % len(PALETTE)], zorder=1)
        for w in r["walls"]:
            a, b = np.array(w["from"]), np.array(w["to"])
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#222" if w["ends_measured"] else "#999",
                    lw=2.5 if w["plane_measured"] else 1.2, zorder=3)
            L = w["length_m"]
            if L["value"] >= 0.4:
                m = (a + b) / 2
                d = b - a
                nrm = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6)
                # put the label inside the room
                if not _inside(V, m + 0.05 * nrm):
                    nrm = -nrm
                p = m + 0.22 * nrm
                rot = 0 if abs(d[0]) > abs(d[1]) else 90
                half = (L["ci95"][1] - L["ci95"][0]) / 2
                ax.text(p[0], p[1], f"{L['value']:.2f}±{half:.2f}", ha="center", va="center",
                        rotation=rot, fontsize=7, zorder=5)
        c = V.mean(0)
        A = r["floor_area_m2"]
        H = r["ceiling_height_m"]
        htxt = f"h {H['value']:.2f} m" if H["value"] is not None else "h n/a"
        ax.text(c[0], c[1], f"{r.get('name', r['id'])}\n{A['value']:.1f} m²\n{htxt}", ha="center",
                va="center", fontsize=9, weight="bold", zorder=6)
    for o in plan.get("openings", []):
        lo, hi = o["span"]
        if o["axis"] == "x":
            xs, zs = [o["coord"]] * 2, [lo, hi]
        else:
            xs, zs = [lo, hi], [o["coord"]] * 2
        ax.plot(xs, zs, "-", color="#c0392b", lw=5, solid_capstyle="butt", zorder=4)
        ax.text(np.mean(xs), np.mean(zs), f"{o['width_m']['value']:.2f}", color="#c0392b", fontsize=7,
                ha="center", va="bottom", zorder=6)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("m"); ax.set_ylabel("m")
    ax.set_title(title or "Floor plan (95% intervals)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _inside(V, p):
    from matplotlib.path import Path
    return Path(V).contains_point(p)
