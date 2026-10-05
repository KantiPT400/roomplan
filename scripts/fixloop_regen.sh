#!/usr/bin/env bash
# Regenerate the fix-loop before and after runs from raw inputs.
#   bash scripts/fixloop_regen.sh <stray_scanner_floor_only_capture> <work_dir>
# before = commit 1c92fa8 (photo tier before the fix), after = the "Fix loop:" commit (this checkout).
set -euo pipefail
CAP=$(realpath "$1"); WORK=$(realpath -m "$2"); REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$WORK"
# 1. the photo set, exactly as benchmarked (frame numbers from the committed manifest)
python "$REPO/scripts/make_photo_set.py" "$CAP" none "$WORK/photos" --manifest "$REPO/docs/benchmark/floor_only_photos_manifest.json"
# 2. LiDAR reference of the same capture
(cd "$REPO" && python -m roomplan "$CAP" --out "$WORK/runs_ref" --no-damage)
# 3. before: photo tier at 1c92fa8 in a separate worktree
git -C "$REPO" worktree add -f "$WORK/before_src" 1c92fa8 >/dev/null 2>&1 || true
mkdir -p "$WORK/before_src/models" && ln -sf "$REPO/models/depth_anything_v2_vits.onnx" "$WORK/before_src/models/" 
(cd "$WORK/before_src" && python -m roomplan "$WORK/photos" --out "$WORK/runs_before" --tier photo --no-damage)
# 4. after: photo tier at this checkout
(cd "$REPO" && python -m roomplan "$WORK/photos" --out "$WORK/runs_after" --tier photo --no-damage)
# 5. score both with the same (current) benchmark code
for t in before after; do
  mkdir -p "$WORK/score_$t"
  ln -sfn "$WORK/runs_ref/$(basename "$CAP")_lidar" "$WORK/score_$t/floor_only_lidar"
  ln -sfn "$WORK/runs_$t/photos_photo" "$WORK/score_$t/floor_only_photos_photo"
  python "$REPO/scripts/benchmark.py" --runs "$WORK/score_$t" --out "$WORK/bench" --tag "$t"
done
# 6. registration truth (every photo has the capture's own pose)
python "$REPO/scripts/photo_registration_truth.py" "$WORK/photos" "$CAP" "$WORK/runs_after/photos_photo/work" "$WORK/bench/edges_after.csv"
git -C "$REPO" show 1c92fa8:roomplan/mvreg.py > "$WORK/mvreg_before.py"
python "$REPO/scripts/photo_registration_truth.py" "$WORK/photos" "$CAP" "$WORK/runs_after/photos_photo/work" "$WORK/bench/edges_before.csv" "$WORK/mvreg_before.py"
