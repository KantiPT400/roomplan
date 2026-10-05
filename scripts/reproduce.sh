#!/usr/bin/env bash
# Regenerate every number in docs/report.md from raw inputs.
#
#   bash scripts/reproduce.sh /path/to/sample_data [out_dir]
#
# sample_data must contain the three supplied Stray Scanner captures:
#   single_room/<id>/  single_scan_floor_only/<id>/  single_scan_with_ceiling/<id>/
# (each with rgb.mp4, depth/, odometry.csv, camera_matrix.csv). Runtime on a 2-core CPU: ~45-60 min.
set -euo pipefail
SAMPLE=$(realpath "$1"); OUT=$(realpath -m "${2:-out}")
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
bash scripts/fetch_models.sh
DATA="$OUT/data"; mkdir -p "$DATA" "$OUT/runs" "$OUT/benchmark"

inner() { dirname "$(find -L "$1" -maxdepth 2 -name odometry.csv | head -1)"; }
ln -sfn "$(inner "$SAMPLE/single_room")" "$DATA/room"
ln -sfn "$(inner "$SAMPLE/single_scan_floor_only")" "$DATA/floor_only"
ln -sfn "$(inner "$SAMPLE/single_scan_with_ceiling")" "$DATA/with_ceiling"

echo "== LiDAR tier (drift on), damage on"
for c in room floor_only with_ceiling; do python -m roomplan "$DATA/$c" --out "$OUT/runs"; done
echo "== LiDAR drift ablation (off)"
python -m roomplan "$DATA/with_ceiling" --out "$OUT/runs" --drift off --no-damage
python scripts/drift_ablation.py "$DATA/with_ceiling" "$OUT/benchmark"

echo "== Photo tier: folders cut from floor_only (exact frames from the committed manifest)"
python scripts/make_photo_set.py "$DATA/floor_only" none "$DATA/floor_only_photos" \
    --manifest docs/benchmark/floor_only_photos_manifest.json
python -m roomplan "$DATA/floor_only_photos" --out "$OUT/runs" --tier photo --no-damage

echo "== Video tier: the single_room clip alone"
mkdir -p "$DATA/room_video"; ln -sfn "$DATA/room/rgb.mp4" "$DATA/room_video/rgb.mp4"
python -m roomplan "$DATA/room_video" --out "$OUT/runs" --tier video --no-damage

echo "== Staged damage (synthetic stain + crack painted into single_room frames)"
python scripts/inject_damage.py "$DATA/room" "$OUT/inject_room"

echo "== Synthetic ground truth (6 rendered captures)"
python scripts/synth_benchmark.py --seeds 0 1 2 3 4 5 --out "$OUT/synth" --data "$DATA"

echo "== Benchmark (repeatability, calibration, tiers vs LiDAR, timing)"
python scripts/benchmark.py --runs "$OUT/runs" --out "$OUT/benchmark" --tag reproduced
ROOMPLAN_OUT="$OUT/runs" python -m pytest -q tests/ || true
echo "done: $OUT/benchmark/benchmark_reproduced.json, $OUT/synth/synth_benchmark.json, $OUT/inject_room/inject_summary.json"
