# Fix loop: result and post-mortem

Declaration: [`fix_loop_declaration.md`](fix_loop_declaration.md) (committed in `b7fe7ff`, evidence corrected in
`a5f5f79`, both before the fix was run). Before = `1c92fa8`. After = the commit that adds this file.
Regenerate both from raw inputs: `bash scripts/fixloop_regen.sh <single_scan_floor_only capture> <work_dir>`.
Readable diff of the fix: `git diff 1c92fa8 <after> -- roomplan/mvreg.py`.

## What shipped
1. Features on the floor are not matched: each photo's floor (pixels within 15 cm of its single-view floor
   plane; bottom 35% of the image for photos with no floor estimate) is masked before SIFT.
2. A registration is rejected when the depths it predicts for the second photo disagree with that photo's own
   single-view depth (median ratio outside 0.7-1.4).
3. Crash guard (not part of the fix, needed to run it): only the largest connected component is mapped; the
   earlier side-by-side placement of disconnected components produced "sites" hundreds of metres wide whose
   grids exhausted memory and restarted the machine twice.

## Numbers (same photo set, same benchmark code)

| metric | before | predicted | after |
|---|---|---|---|
| accepted registrations within 5 deg of the true pose | 44 / 203 (22%) | >= 60% | 34 / 109 (31%) |
| registrations off by > 15 deg | 157 (77%) | | 67 (61%) |
| global solve: median heading residual over all edges | 36.5 deg | | **3.3 deg** |
| global solve: median position residual | 0.23 m | | **0.09 m** |
| photos in the mapped component | 52 / 55 (scattered over ~30 m) | | 43 / 55 (one ~11 x 8 m site) |
| LiDAR rooms matched by a photo-tier room | 0 / 8 | >= 4 | 1 / 8 |
| footprint ratio to LiDAR | 0.543 | 0.80-1.20 | 0.523 |
| gate: one plan, correct adjacency, footprint +-8% | fail | fail (expected) | **fail** |

## Verdict
The root cause was real and the fix removed it where it acted: half of the false registrations are gone and the
layout went from self-contradictory (36.5 deg median disagreement between edges) to consistent (3.3 deg,
9 cm). **The predictions for the gate metrics were badly wrong**: rooms matched 1 (predicted >= 4) and the
footprint ratio did not move (0.52, predicted 0.80-1.20).

## Post-mortem: why the prediction failed
1. **A second source of false registrations we had not looked for.** 67 false edges remain; their inliers are
   mostly not on the floor any more (32% vs 71% before), and 6 of the 8 strongest come from one photo
   (`Room_1/IMG_01196`, 68-77 inliers each), matched to photos in its own room and in three other rooms,
   1.2-6.8 m away: a repetitive pattern above floor level acting as a hub. Floor masking cannot remove that. Hub suppression
   (a photo that "registers" to many photos in different folders with mutually inconsistent poses) and a
   higher inlier floor (>= 80 inliers: 79% correct) are the next fix.
2. **The gate needs walls, and mono-depth walls rarely pass the wall test.** Even with a consistent layout,
   almost every room edge in the photo plan is unsupported (grey in `docs/benchmark/fixloop_after_photo_plan.png`).
   The LiDAR wall test asks a 4 cm cell to be occupied over 60% of floor-to-ceiling height; photos taken at chest
   height pitched down see the lower walls, and mono depth smears each wall over ~10 cm, so rooms do not
   separate and the footprint stays partial. We predicted the footprint from the registration fix alone and did
   not check this second stage before predicting. That is the error in our reasoning.
3. **The photo set itself.** It is cut from a walkthrough shot for LiDAR, not from photos shot for overlap, so
   few photo pairs truly overlap. A protocol-following capture (docs/capture_protocol.md, section C) should
   do better, but we have no such capture to prove it.
