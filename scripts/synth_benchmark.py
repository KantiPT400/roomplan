"""Accuracy against exact ground truth on rendered captures (scripts/synth_capture.py).

    python scripts/synth_benchmark.py --seeds 0 1 2 3 --out out/synth

For every seed: render (if missing), run the LiDAR tier with drift correction, then score each reported
quantity against the truth and check whether its 95% interval contains the truth (calibration).
Rooms: the plan room containing the truth room's centre camera path is matched to it; a dimension is scored
with the room edge that spans >= 95% of the room's bounding-box side along that axis.
"""
import argparse, json, os, subprocess, sys
import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out", default="out/synth")
ap.add_argument("--data", default="data")
ap.add_argument("--frames", type=int, default=1200)
a = ap.parse_args()
rows = []
for sd in a.seeds:
    cap = os.path.join(a.data, f"synth_s{sd}")
    if not os.path.exists(os.path.join(cap, "ground_truth.json")):
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "synth_capture.py"), cap, "--seed", str(sd),
                        "--frames", str(a.frames)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", "roomplan", os.path.abspath(cap), "--out", os.path.abspath(a.out), "--no-damage"],
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    gt = json.load(open(os.path.join(cap, "ground_truth.json")))
    plan = json.load(open(os.path.join(a.out, f"synth_s{sd}_lidar", "plan.json")))
    rooms = sorted(plan["rooms"], key=lambda r: -r["floor_area_m2"]["value"])
    rec = {"seed": sd, "rooms_found": len(plan["rooms"]), "openings_found": len(plan["openings"]),
           "adjacency_ok": len(plan["adjacency"]) == 1}
    # truth A is the larger room, B the smaller; the plan's two largest rooms are matched by size order
    for name, r in zip(("A", "B"), rooms[:2]):
        V = np.array(r["polygon_m"]); ext = V.max(0) - V.min(0)
        truth = sorted(gt["rooms"][name]["dims_m"])
        for k, (axis_len) in enumerate(sorted(ext)):
            ax = int(np.argsort(ext)[k])
            # edges parallel to this axis spanning the side
            best = None
            for w in r["walls"]:
                d = np.abs(np.array(w["to"]) - np.array(w["from"]))
                if d[ax] >= 0.95 * ext[ax] and (best is None or w["length_m"]["sigma"] < best["length_m"]["sigma"]):
                    best = w
            t = truth[k]
            if best is None:
                rec[f"{name}_dim{k}_err"] = None
                continue
            L = best["length_m"]
            rec[f"{name}_dim{k}_err"] = round(L["value"] - t, 4)
            rec[f"{name}_dim{k}_in_ci"] = bool(L["ci95"][0] <= t <= L["ci95"][1])
            rec[f"{name}_dim{k}_halfwidth"] = round((L["ci95"][1] - L["ci95"][0]) / 2, 4)
        if "dimensions_m" in r:                     # wall-to-wall (laser-measure equivalent)
            dd = sorted([r["dimensions_m"]["along_x"], r["dimensions_m"]["along_z"]], key=lambda m: m["value"])
            for k, (m, t) in enumerate(zip(dd, truth)):
                rec[f"{name}_ww{k}_err"] = round(m["value"] - t, 4)
                rec[f"{name}_ww{k}_in_ci"] = bool(m["ci95"][0] <= t <= m["ci95"][1])
        H = r["ceiling_height_m"]
        if H["value"] is not None:
            rec[f"{name}_ceil_err"] = round(H["value"] - gt["rooms"][name]["ceiling_m"], 4)
            rec[f"{name}_ceil_in_ci"] = bool(H["ci95"][0] <= gt["rooms"][name]["ceiling_m"] <= H["ci95"][1])
    doors = [o for o in plan["openings"] if len(o["rooms"]) == 2]
    if doors:
        o = min(doors, key=lambda o: abs(o["width_m"]["value"] - gt["door"]["width_m"]))
        rec["door_err"] = round(o["width_m"]["value"] - gt["door"]["width_m"], 4)
        rec["door_in_ci"] = bool(o["width_m"]["ci95"][0] <= gt["door"]["width_m"] <= o["width_m"]["ci95"][1])
    rec["drift"] = {k: plan["meta"]["drift"].get(k) for k in ("max_correction_m", "yaw_range_deg")}
    rec["true_drift_final"] = gt.get("drift_final")
    rows.append(rec)
    print(json.dumps(rec))
os.makedirs(a.out, exist_ok=True)
summary = {"seeds": a.seeds, "rows": rows}
def coll(prefix, suffix):
    return [r[k] for r in rows for k in r if k.startswith(prefix) and k.endswith(suffix) and r[k] is not None]
errs = np.array([abs(x) for x in coll("A_dim", "_err") + coll("B_dim", "_err")])
inci = coll("A_dim", "_in_ci") + coll("B_dim", "_in_ci")
summary["walls"] = {"n": len(errs), "median_abs_err_m": float(np.median(errs)) if len(errs) else None,
                    "max_abs_err_m": float(errs.max()) if len(errs) else None,
                    "within_1cm": int((errs <= 0.01).sum()), "within_2cm": int((errs <= 0.02).sum()),
                    "ci95_coverage": float(np.mean(inci)) if inci else None}
ww = np.array([abs(x) for x in coll("A_ww", "_err") + coll("B_ww", "_err")])
summary["wall_to_wall"] = {"n": len(ww), "median_abs_err_m": float(np.median(ww)) if len(ww) else None,
                           "max_abs_err_m": float(ww.max()) if len(ww) else None,
                           "within_1cm": int((ww <= 0.01).sum()), "within_2cm": int((ww <= 0.02).sum()),
                           "ci95_coverage": float(np.mean(coll("A_ww", "_in_ci") + coll("B_ww", "_in_ci"))) if len(ww) else None}
ce = np.array([abs(x) for x in coll("A_ceil", "_err") + coll("B_ceil", "_err")])
summary["ceiling"] = {"n": len(ce), "max_abs_err_m": float(ce.max()) if len(ce) else None,
                      "within_1.5cm": int((ce <= 0.015).sum()),
                      "ci95_coverage": float(np.mean(coll("A_ceil", "_in_ci") + coll("B_ceil", "_in_ci")))}
de = np.array([abs(r["door_err"]) for r in rows if "door_err" in r])
summary["door"] = {"found": len(de), "of": len(rows), "abs_err_m": de.round(4).tolist(),
                   "within_2cm": int((de <= 0.02).sum()), "ci95_coverage": float(np.mean([r["door_in_ci"] for r in rows if "door_in_ci" in r])) if len(de) else None}
summary["rooms_found"] = [r["rooms_found"] for r in rows]
json.dump(summary, open(os.path.join(a.out, "synth_benchmark.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
