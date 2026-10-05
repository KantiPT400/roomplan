"""One command per capture:

    python -m roomplan <capture_dir> [--out out/] [--tier auto|lidar|video|photo] [--drift on|off]

Tier is detected from the folder: depth/ + odometry.csv -> lidar; a video file -> video;
sub-folders of images (one per room) -> photo.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time


def detect_tier(path):
    if os.path.isdir(os.path.join(path, "depth")) and os.path.exists(os.path.join(path, "odometry.csv")):
        return "lidar"
    exts_v = (".mp4", ".mov", ".m4v")
    if os.path.isfile(path) and path.lower().endswith(exts_v):
        return "video"
    if os.path.isdir(path):
        files = os.listdir(path)
        if any(f.lower().endswith(exts_v) for f in files):
            return "video"
        subs = [d for d in files if os.path.isdir(os.path.join(path, d))]
        if subs:
            return "photo"
    raise SystemExit(f"cannot tell what kind of capture {path} is")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roomplan")
    ap.add_argument("capture")
    ap.add_argument("--out", default="out")
    ap.add_argument("--tier", default="auto", choices=["auto", "lidar", "video", "photo"])
    ap.add_argument("--drift", default="on", choices=["on", "off"])
    ap.add_argument("--stride", type=int, default=10, help="use every Nth depth frame (lidar)")
    ap.add_argument("--no-damage", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="video/photo: reuse the registration of a previous run (geometry only)")
    a = ap.parse_args(argv)
    tier = detect_tier(a.capture) if a.tier == "auto" else a.tier
    if a.reuse:
        os.environ["ROOMPLAN_REUSE"] = "1"
    name = os.path.basename(os.path.normpath(a.capture))
    out = os.path.join(a.out, f"{name}_{tier}" + ("" if a.drift == "on" else "_driftoff"))
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if tier == "lidar":
        from .lidar import run
        plan, dbg = run(a.capture, stride=a.stride, drift=a.drift)
    elif tier == "video":
        from .video import run
        plan, dbg = run(a.capture, out)
    else:
        from .photo import run
        plan, dbg = run(a.capture, out)
    from .contract import finalize
    plan = finalize(plan, capture=name, tier=tier, dbg=dbg, capture_dir=a.capture, out_dir=out,
                    damage=not a.no_damage)
    plan["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    from .render import render
    import numpy as np
    pts = dbg.get("plan_points")
    render(plan, os.path.join(out, "plan.png"), title=f"{name} - {tier} tier (95% intervals)", points=pts)
    print(f"{name}: {len(plan['rooms'])} rooms, {len(plan['openings'])} openings, "
          f"{len(plan['damage'])} damage regions -> {out}/plan.json ({plan['meta']['runtime_s']} s)")


if __name__ == "__main__":
    main()
