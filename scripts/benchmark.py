"""Regenerate every reported benchmark number from the plan.json files of a run.

    python scripts/benchmark.py --runs out/ --out out/benchmark [--tag before]

Inputs are the outputs of `python -m roomplan` on the sample captures (see README, "Reproduce").
No ground truth existed for the sample data, so every number here is one of:
  * repeatability : two independent LiDAR captures of the same apartment (floor_only vs with_ceiling,
                    shown to be the same property by plan registration: correlation 0.55-0.61 at 90 deg vs
                    <= 0.28 at the other rotations);
  * calibration   : z = (difference between captures) / sqrt(sigma_a^2 + sigma_b^2). If the 95% intervals
                    are honest, |z| <= 1.96 for ~95% of walls. This needs no ground truth;
  * reference     : video/photo tier vs the LiDAR tier of the SAME capture (LiDAR is the reference, itself
                    unverified against tape);
  * drift         : loop-closure residual and wall sharpness with drift correction on vs off.
"""
import argparse
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from roomplan.evaluate import compare, align, transform, match_rooms


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def repeatability(A, B):
    r = compare(A, B)
    W = r["walls"]
    d = np.array([w["diff_m"] for w in W]) if W else np.array([])
    z = np.array([w["z"] for w in W]) if W else np.array([])
    out = {
        "alignment_score": round(r["alignment"]["score"], 3), "rotation_deg": r["alignment"]["rot_deg"],
        "rooms_matched": len(r["rooms"]), "rooms_a": len(A["rooms"]), "rooms_b": len(B["rooms"]),
        "walls_compared": len(W),
        "walls_pass_1cm_or_0.5pct": int(sum(w["pass_repeat"] for w in W)),
        "pass_rate": round(float(np.mean([w["pass_repeat"] for w in W])), 3) if W else None,
        "median_abs_diff_m": round(float(np.median(np.abs(d))), 4) if len(d) else None,
        "p90_abs_diff_m": round(float(np.percentile(np.abs(d), 90)), 4) if len(d) else None,
        "calibration_frac_within_95ci": round(float(np.mean(np.abs(z) <= 1.96)), 3) if len(z) else None,
        "rooms": r["rooms"], "walls": W,
    }
    return out


def plane_repeatability(A, B):
    """Wall-plane positions and plane-to-plane spans, independent of how rooms were segmented."""
    from roomplan.evaluate import _wall_axis
    al = align(A, B)
    Bt = transform(B, al["M"], al["t"])
    def planes(p):
        o = []
        for rm in p["rooms"]:
            for w in rm["walls"]:
                if w["plane_measured"]:
                    o.append(_wall_axis(w, rm["polygon_m"]))
        return o
    pa, pb = planes(A), planes(Bt)
    pairs = []
    for ax, c, s, fa in pa:
        best = None
        for bx, cb, sb, fb in pb:
            if bx != ax or fb != fa or abs(cb - c) > 0.15:
                continue
            ov = min(s[1], sb[1]) - max(s[0], sb[0])
            if ov > 0.4 and (best is None or abs(cb - c) < abs(best - c)):
                best = cb
        if best is not None:
            pairs.append((ax, c, best, s))
    spans = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = pairs[i], pairs[j]
            if a[0] != b[0]:
                continue
            ov = min(a[3][1], b[3][1]) - max(a[3][0], b[3][0])
            da, db = abs(a[1] - b[1]), abs(a[2] - b[2])
            if 0.8 < da < 6 and ov > 0.5:
                spans.append((da, db))
    S = np.array(spans) if spans else np.zeros((0, 2))
    diff = S[:, 1] - S[:, 0] if len(S) else np.array([])
    ok = (np.abs(diff) <= 0.01) | (np.abs(diff) <= 0.005 * S[:, 0]) if len(S) else np.array([])
    pos = np.array([p[2] - p[1] for p in pairs])
    # relative scale between the captures: slope of plane positions B vs A per axis (1.0 = no scale error)
    slope = {}
    for ax_ in ("x", "z"):
        P = np.array([(p[1], p[2]) for p in pairs if p[0] == ax_])
        if len(P) > 2 and np.ptp(P[:, 0]) > 1.0:
            slope[ax_] = round(float(np.polyfit(P[:, 0], P[:, 1], 1)[0]), 4)
    return {"planes_matched": len(pairs), "relative_scale_slope": slope,
            "plane_position_median_abs_m": round(float(np.median(np.abs(pos))), 4) if len(pos) else None,
            "planes_within_1cm": int((np.abs(pos) <= 0.01).sum()), "planes_within_2cm": int((np.abs(pos) <= 0.02).sum()),
            "spans_compared": len(S), "spans_pass": int(ok.sum()) if len(S) else 0,
            "span_pass_rate": round(float(ok.mean()), 3) if len(S) else None,
            "span_median_abs_diff_m": round(float(np.median(np.abs(diff))), 4) if len(diff) else None}


def _opening_segs(plan, M=None, t=None):
    out = []
    for o in plan["openings"]:
        lo, hi = o["span"]
        a, b = ((o["coord"], lo), (o["coord"], hi)) if o["axis"] == "x" else ((lo, o["coord"]), (hi, o["coord"]))
        a, b = np.array(a, float), np.array(b, float)
        if M is not None:
            a, b = a @ M.T + t, b @ M.T + t
        out.append((o, a, b))
    return out


def opening_repeatability(A, B, tol_perp=0.15, tol_along=0.25):
    """Door/passage widths measured twice (two captures of the same property). This is the only real-data
    evidence for the opening gate (<= 2 cm on >= 85%) we can produce without a tape measure: two captures
    agreeing is necessary, not sufficient, for both being right."""
    al = align(A, B)
    sa, sb = _opening_segs(A), _opening_segs(B, al["M"], al["t"])
    pairs, used = [], set()
    for oa, a0, a1 in sa:
        da = a1 - a0; ca = (a0 + a1) / 2; ua = da / (np.linalg.norm(da) + 1e-9)
        best = None
        for j, (ob, b0, b1) in enumerate(sb):
            if j in used:
                continue
            db = b1 - b0; cb = (b0 + b1) / 2
            if abs(np.dot(ua, db / (np.linalg.norm(db) + 1e-9))) < 0.9:      # not parallel
                continue
            off = cb - ca
            along, perp = abs(np.dot(off, ua)), abs(ua[0] * off[1] - ua[1] * off[0])
            if perp < tol_perp and along < tol_along and (best is None or along < best[0]):
                best = (along, j, ob)
        if best is not None:
            used.add(best[1]); ob = best[2]
            wa, wb = oa["width_m"], ob["width_m"]
            d = wb["value"] - wa["value"]
            s = np.hypot(wa["sigma"] or 0, wb["sigma"] or 0)
            pairs.append({"a": oa["id"], "b": ob["id"], "kind_a": oa["kind"], "kind_b": ob["kind"],
                          "width_a": wa["value"], "width_b": wb["value"], "diff_m": round(d, 4),
                          "z": round(d / s, 2) if s > 0 else None})
    d = np.array([p["diff_m"] for p in pairs])
    z = np.array([p["z"] for p in pairs if p["z"] is not None])
    return {"openings_a": len(sa), "openings_b": len(sb), "matched": len(pairs),
            "within_2cm": int((np.abs(d) <= 0.02).sum()) if len(d) else 0,
            "within_2cm_frac": round(float(np.mean(np.abs(d) <= 0.02)), 3) if len(d) else None,
            "median_abs_diff_m": round(float(np.median(np.abs(d))), 4) if len(d) else None,
            "calibration_frac_within_95ci": round(float(np.mean(np.abs(z) <= 1.96)), 3) if len(z) else None,
            "pairs": pairs}


def reference(R, T):
    """Tier T against reference plan R (same capture)."""
    r = compare(R, T)
    W = r["walls"]
    rel = np.array([abs(w["rel"]) for w in W]) if W else np.array([])
    inside = [w["a_in_ci_b"] for w in W]
    aR = sum(x["floor_area_m2"]["value"] for x in R["rooms"])
    aT = sum(x["floor_area_m2"]["value"] for x in T["rooms"])
    return {"rooms_ref": len(R["rooms"]), "rooms_tier": len(T["rooms"]), "rooms_matched": len(r["rooms"]),
            "footprint_ratio": round(aT / aR, 3) if aR else None,
            "walls_compared": len(W),
            "walls_within_8pct": int((rel <= 0.08).sum()) if len(rel) else 0,
            "walls_within_3pct": int((rel <= 0.03).sum()) if len(rel) else 0,
            "median_rel_err": round(float(np.median(rel)), 3) if len(rel) else None,
            "ref_inside_tier_ci95": round(float(np.mean(inside)), 3) if inside else None,
            "alignment_score": round(r["alignment"]["score"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="out")
    ap.add_argument("--out", default="out/benchmark")
    ap.add_argument("--tag", default="current")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = lambda n: load(os.path.join(a.runs, n, "plan.json"))
    res = {"tag": a.tag}
    wc, fo = P("with_ceiling_lidar"), P("floor_only_lidar")
    if wc and fo:
        res["repeatability_lidar"] = repeatability(wc, fo)
        res["plane_repeatability_lidar"] = plane_repeatability(wc, fo)
        res["opening_repeatability_lidar"] = opening_repeatability(wc, fo)
    # ceilings
    res["ceilings"] = {}
    for n in ("with_ceiling_lidar", "floor_only_lidar", "room_lidar"):
        p = P(n)
        if p:
            res["ceilings"][n] = [{"room": r["id"], **{k: r["ceiling_height_m"].get(k) for k in
                                                       ("value", "ci95", "status", "coverage")}} for r in p["rooms"]]
    # drift ablation
    on, off = P("with_ceiling_lidar"), P("with_ceiling_lidar_driftoff")
    if on and off:
        res["drift"] = {"on": on["meta"]["drift"], "off_rooms": len(off["rooms"]), "on_rooms": len(on["rooms"])}
    # openings
    res["openings"] = {n: [{"kind": o["kind"], "width": o["width_m"]["value"], "ci95": o["width_m"]["ci95"],
                            "rooms": o["rooms"]} for o in P(n)["openings"]]
                       for n in ("with_ceiling_lidar", "floor_only_lidar", "room_lidar") if P(n)}
    # tiers vs LiDAR reference of the same capture
    res["reference"] = {}
    for tier_run, ref_run in (("floor_only_photos_photo", "floor_only_lidar"),
                              ("floor_only_video_video", "floor_only_lidar"),
                              ("room_video_video", "room_lidar")):
        T, R = P(tier_run), P(ref_run)
        if T and R:
            try:
                res["reference"][tier_run] = reference(R, T)
                res["reference"][tier_run]["meta"] = {k: T["meta"].get(k) for k in
                                                      ("photos_total", "photos_placed", "room_folders_stitched",
                                                       "room_folders_unstitched", "sfm_fragments",
                                                       "frames_registered", "runtime_s")}
            except Exception as e:                       # a tier output too broken to even align
                res["reference"][tier_run] = {"error": repr(e)}
    res["timing_s"] = {os.path.basename(os.path.dirname(f)): load(f)["meta"].get("runtime_s")
                       for f in glob.glob(os.path.join(a.runs, "*", "plan.json"))}
    json.dump(res, open(os.path.join(a.out, f"benchmark_{a.tag}.json"), "w"), indent=1, default=float)
    # short console summary
    r = res.get("repeatability_lidar", {})
    print(f"[{a.tag}] repeatability: {r.get('walls_pass_1cm_or_0.5pct')}/{r.get('walls_compared')} walls "
          f"(median |d| {r.get('median_abs_diff_m')} m), calibration {r.get('calibration_frac_within_95ci')}; "
          f"planes: {res.get('plane_repeatability_lidar')}")
    for k, v in res["reference"].items():
        print(" ", k, {x: v.get(x) for x in ("rooms_matched", "footprint_ratio", "walls_compared",
                                              "walls_within_8pct", "median_rel_err", "ref_inside_tier_ci95", "error")})


if __name__ == "__main__":
    main()
