# Fix declaration (written and committed BEFORE the fix; the fix and the after run come in later commits)

## 1. Worst gate in our own benchmark: photo-tier whole-property stitch

Gate: per-room photo folders -> one stitched plan, correct adjacency, no overlaps, footprint within +-8%.

Before run: photo tier at commit `1c92fa8` on `floor_only_photos` (55 photos, 7 room folders, cut from the
`single_scan_floor_only` capture by `scripts/make_photo_set.py`), compared with the LiDAR tier of the same
capture (`scripts/benchmark.py --tag before`, file `out/benchmark/benchmark_before.json`):

| metric | before |
|---|---|
| stitched footprint / LiDAR footprint | **0.543** (-46%; gate +-8%) |
| LiDAR rooms matched by a photo-tier room (IoU >= 0.4) | **0 of 8** |
| rooms in the photo-tier plan | 1 |
| photos joined into the largest component | 52 of 55 (yet laid out over ~30 m) |

## 2. Root-cause hypothesis and evidence

**Hypothesis: most pairwise registrations are false, caused by the repetitive floor tiles; the global
solve cannot recover because false edges outnumber true ones ~11:1.**

Evidence (`scripts/photo_registration_truth.py`; every photo has a true ARKit pose because the set was cut
from a LiDAR capture; file `out/benchmark/photo_edges_truth_before.csv`):
- 203 registrations were accepted; **186 are off by > 15 deg** (median rotation error 112 deg); only 7 are
  within 5 deg.
- inliers of the false registrations lie on the floor (lower 40% of the image) **61%** of the time vs **9%**
  for the true ones;
- false registrations join photos a median **3.98 m** apart (they cannot overlap); true ones 0.03 m;
- false registrations cross room folders 72% of the time; true ones 29%;
- false registrations have a median of 50 inliers vs 209: the 25-inlier acceptance threshold lets a tile
  grid through, and an essential matrix fitted to a near-planar pattern is degenerate.

## 3. Fix we will ship, and the predicted numbers

- Do not use features on the floor: each photo's single-view geometry (singleview.py) already gives its
  floor plane; mask those pixels before SIFT.
- Reject a registration when the depths it predicts for the second photo disagree with that photo's own
  single-view depth by more than x0.7-x1.4 (median ratio).
- Keep the existing gravity-consistency check.

Predicted after the fix (same photo set, same benchmark script):
- >= 60% of accepted registrations within 5 deg of truth (before: 3%);
- >= 4 of 8 LiDAR rooms matched by photo-tier rooms (before: 0);
- footprint ratio within 0.80-1.20 (before: 0.543).
We expect the +-8% footprint gate itself to remain at risk: metric scale in this tier comes from the
1.40 +- 0.07 m camera-height prior (5% at 1 sigma), and the photo set is cut from a walkthrough that was not
shot for overlapping stills.

## Process note
Commit `b7fe7ff` (this declaration) accidentally also contained the fix's code in `roomplan/mvreg.py`
(a `git stash` of it failed silently). The code had not been run when the declaration was written and no
after-numbers existed. Commit after it backs the code out again so that the fix lands in its own commit,
together with its after run: `before` = `1c92fa8`, `after` = the "Fix loop: ..." commit.
