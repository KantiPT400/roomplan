# roomplan: floor plans, measurements and wall damage from a phone capture

One command per capture:

```bash
python -m roomplan <capture_folder>            # tier detected from the folder
python -m roomplan <capture_folder> --tier lidar|video|photo --out out/
```

Writes `out/<capture>_<tier>/plan.json` (schema: `roomplan/schema.json`) and `plan.png`: every room with
walls, floor area, ceiling height and openings, the stitched whole-property plan with adjacency, wall damage
regions with class and metric extent, concealed-damage flags with the rule that fired, scope line items keyed
to surfaces, and a 95% interval on every measurement.

| Capture folder contains | Tier | Capture with |
|---|---|---|
| `depth/`, `odometry.csv`, `rgb.mp4`, ... | LiDAR | Stray Scanner app (iPhone/iPad Pro) |
| one `.mp4`/`.mov` | video | iPhone Camera app |
| one sub-folder of photos per room | photo | iPhone Camera app |

How to capture: [`docs/capture_protocol.md`](docs/capture_protocol.md) (one page, written for a non-engineer).

## Setup (clean machine, ~10 minutes)

Python 3.10-3.13 and `ffmpeg` on the PATH.

```bash
git clone https://github.com/KantiPT400/roomplan && cd roomplan
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/fetch_models.sh                           # Depth Anything V2 small (ONNX, 99 MB), checksum-verified
python -m roomplan /path/to/stray_scanner_export       # e.g. the sample single_room/c00a170fe1
```

Everything runs on CPU, offline once the weights are fetched; nothing calls our infrastructure.

## Reproduce every reported number

```bash
bash scripts/reproduce.sh /path/to/sample_data         # folders single_room/, single_scan_floor_only/, single_scan_with_ceiling/
```

See [`docs/report.md`](docs/report.md) for what each number means and its limits. Fix loop:
[`docs/fix_loop_declaration.md`](docs/fix_loop_declaration.md) (declaration, committed before the fix) and
[`docs/fix_loop_result.md`](docs/fix_loop_result.md).

## Repository map

| Path | What |
|---|---|
| `roomplan/lidar.py` | LiDAR tier: fuse depth, floor/ceiling, walls, rooms, openings, intervals |
| `roomplan/drift.py` | drift accountability: plane-anchored pose graph with loop closures (`--drift off` = ablation) |
| `roomplan/walls.py`, `layout.py`, `polygon.py` | wall planes, rooms/openings/adjacency, wall-snapped room polygons |
| `roomplan/singleview.py`, `mvreg.py`, `photo.py` | photo tier: per-photo metric depth, pairwise registration, global layout |
| `roomplan/video.py`, `sfm.py`, `pseudo.py` | video tier |
| `roomplan/damage.py`, `scope.py` | wall damage, concealed-damage rules, scope line items |
| `roomplan/uncertainty.py` | error model behind every interval |
| `scripts/benchmark.py` | all benchmark numbers (GT-free: repeatability, calibration, tier vs LiDAR) |
| `scripts/inject_damage.py` | staged-damage test (known synthetic stain + crack painted consistently into every frame) |
| `scripts/make_photo_set.py` | photo-tier test folders cut from a LiDAR capture's video, following the protocol |
| `scripts/photo_registration_truth.py` | scores photo registrations against the capture's own poses |
| `docs/` | protocol, device matrix, compliance matrix, report, fix loop |
