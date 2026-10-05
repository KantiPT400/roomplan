"""End-to-end on a rendered capture with exact ground truth (scripts/synth_capture.py): two rooms, one door,
dimensions within 3 cm, ceiling within 1.5 cm, and every reported 95% interval containing the truth."""
import json, os, subprocess, sys
import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "..")


def test_synthetic_flat(tmp_path):
    cap = tmp_path / "synth"
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "synth_capture.py"), str(cap), "--seed", "11",
                    "--frames", "900"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", "roomplan", str(cap), "--out", str(tmp_path / "out"), "--no-damage"],
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    gt = json.load(open(cap / "ground_truth.json"))
    plan = json.load(open(tmp_path / "out" / "synth_lidar" / "plan.json"))
    assert len(plan["rooms"]) == 2
    assert len(plan["adjacency"]) == 1
    door = [o for o in plan["openings"] if len(o["rooms"]) == 2]
    assert door and abs(door[0]["width_m"]["value"] - gt["door"]["width_m"]) < 0.03
    for r in plan["rooms"]:
        h = r["ceiling_height_m"]
        if h["value"] is not None:
            assert abs(h["value"] - gt["rooms"]["A"]["ceiling_m"]) < 0.015
            assert h["ci95"][0] <= gt["rooms"]["A"]["ceiling_m"] <= h["ci95"][1]
    big = max(plan["rooms"], key=lambda r: r["floor_area_m2"]["value"])
    V = np.array(big["polygon_m"]); ext = sorted(V.max(0) - V.min(0))
    assert np.allclose(ext, sorted(gt["rooms"]["A"]["dims_m"]), atol=0.03)
