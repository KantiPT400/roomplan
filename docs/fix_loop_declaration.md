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
solve cannot recover because false edges outnumber true ones ~3.5:1.**

Evidence (`scripts/photo_registration_truth.py`; every photo has a true pose because the set was cut from a
LiDAR capture; file `out/benchmark/photo_edges_truth_before.csv`, scored with the before code `1c92fa8`):

| | false registrations (> 15 deg off) | true registrations (< 5 deg) |
|---|---|---|
| count (of 203 accepted) | **157** (median error 121 deg) | 44 |
| share of inliers on the floor (lower 40% of image) | **71%** | 7% |
| photos in the same room folder | 15% | 95% |
| true distance between the two photos | 4.48 m (cannot overlap) | 0.47 m |
| median inliers | 44 | 131 |

The 25-inlier acceptance threshold lets a tile grid through, and an essential matrix fitted to a near-planar
repetitive pattern is degenerate.

**Correction (made before the fix was run or scored):** the first version of this declaration (commit
`b7fe7ff`) said 186 of 203 registrations were false. That came from a bug in the truth checker: it applied an
ARKit y/z axis flip that the Stray Scanner poses do not need (they are already OpenCV camera-to-world, as
`fuse.py` uses them) and ignored the 90 deg rotation that makes the photos upright. Rotation magnitudes
matched the truth within ~1 deg while full rotations did not, which exposed it. Re-scored with the corrected
checker, 157 of 203 are false; the hypothesis and the fix are unchanged, the numbers above are the corrected ones.

## 3. Fix we will ship, and the predicted numbers

- Do not use features on the floor: each photo's single-view geometry (singleview.py) already gives its
  floor plane; mask those pixels before SIFT.
- Reject a registration when the depths it predicts for the second photo disagree with that photo's own
  single-view depth by more than x0.7-x1.4 (median ratio).
- Keep the existing gravity-consistency check.

Predicted after the fix (same photo set, same benchmark script):
- >= 60% of accepted registrations within 5 deg of truth (before: 22%);
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
