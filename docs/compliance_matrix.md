# Compliance matrix

Status: **done** = implemented and evidenced; **partial** = implemented, evidence incomplete or gate not met;
**not done** = missing, with the reason. Numbers: docs/report.md and docs/benchmark/.

## Part 1: capture
| Requirement | File | Artifact | Status |
|---|---|---|---|
| Capture route (Route 2: stock protocol, one page, non-engineer) | docs/capture_protocol.md | protocol naming Stray Scanner / Camera app | done |
| LiDAR tier (depth, poses, intrinsics) | roomplan/lidar.py, io.py, fuse.py | `python -m roomplan <stray_scanner_folder>` | done |
| Video tier (handheld clip, no depth/poses) | roomplan/video.py, mvreg.py, singleview.py | `python -m roomplan <folder with .mp4/.mov>` | partial: runs, plan quality weak on sample |
| Photo tier (2-8 stills per room folder, stitched plan) | roomplan/photo.py, mvreg.py, singleview.py, fallback.py | `python -m roomplan <folder of room folders>` | partial: runs and stitches the connected component; stitch gate fails on sample; folders that cannot be stitched get single-view room estimates (any picture in, results out) |
| Same output contract from every tier; intervals widen with thinner input | roomplan/contract.py, uncertainty.py, pseudo.py | plan.json per tier; video/photo add the 5% scale term | done |
| Device matrix | docs/device_matrix.md | tier x hardware x honest accuracy | done |

## Part 2: output contract
| Requirement | File | Artifact | Status |
|---|---|---|---|
| Per-room plan: walls, ceiling height, floor area, openings | lidar.py, polygon.py, layout.py | `rooms[]`, `openings[]` in plan.json | done |
| Stitched multi-room plan with correct adjacency | layout.py (openings = adjacency edges), render.py | `adjacency[]`, plan.png | done (LiDAR); partial (photo/video) |
| Per-surface damage regions, class, metric extent | roomplan/damage.py | `damage[]` (surface id, class, m^2 with interval), wall elevation images | done (rule-based; staged test) |
| Concealed-damage flags with the rule that fired | damage.py `RULES` | `concealed_damage_flags[]` | done |
| Scope line items keyed to surfaces | roomplan/scope.py | `scope[]` | done |
| Confidence interval on every measurement | uncertainty.py | `{value, ci95, sigma}` everywhere | done |
| One command per capture | roomplan/__main__.py | `python -m roomplan <capture>` | done |
| JSON to a published schema | roomplan/schema.json, tests/test_schema.py | schema + validation test | done |
| Rendered plan | roomplan/render.py | plan.png (docs/sample_outputs/) | done |

## Benchmark set composition
| Requirement | Artifact | Status |
|---|---|---|
| Multi-room capture, 3+ rooms + connector | single_scan_with_ceiling, single_scan_floor_only (supplied) | done |
| Furnished room with staged damage, two classes | scripts/inject_damage.py (synthetic stain + crack painted into real frames) | partial: synthetic staging, no physical room |
| Same rooms at all three tiers, multi-room at photo tier as folders | floor_only LiDAR + photo folders (scripts/make_photo_set.py) + video (rgb.mp4) | partial: photo folders cut from the LiDAR walkthrough |
| A room captured twice at the same tier | with_ceiling vs floor_only (same apartment) | done |
| Laser/tape ground truth on everything | none available; synthetic ground truth instead (scripts/synth_capture.py) | **not done**: no access to the rooms |

## Gates
| Gate | Evidence | Status |
|---|---|---|
| Opening widths <= 2 cm on >= 85%, detection scored | synthetic: 6/6 doors found, 5/6 within 2 cm (0.8-2.2 cm); real: same doors in two captures agree within 2 cm for 2/7 (no tape) | fail on synthetic (83% < 85%); real accuracy not measurable, real agreement 29% |
| Ceiling <= 1.5 cm; spread across captures <= 1 cm | synthetic 12/12 within 2 mm; real spread not computable (one capture saw ceilings) | pass on synthetic; real spread not evaluable |
| Repeatability within 1 cm or 0.5% per wall | real, same apartment twice: 1/8 room walls, 8/45 plane spans, 8/30 planes within 1 cm (median 2.7 cm) | **fail** (cause: surface identity + room partition, report section 5) |
| Drift accountability + on/off ablation | scripts/drift_ablation.py, docs/benchmark/drift_ablation.png | done (loop residual 11.4 -> 2.1 cm) |
| Photo-tier whole-property stitch, +-8% footprint | fix loop + follow-up, docs/fix_loop_result.md | **partial**: footprint 0.958 (passes +-8%), rooms 7 vs 11, adjacency not right |
| Photo +-8% / video +-3% wall lengths; calibration every tier | benchmark.py reference vs LiDAR | fail / not evaluable on sample (too few matched walls) |

## Parts 3-5 and deliverables
| Requirement | File | Status |
|---|---|---|
| Head-to-head vs consumer app on 2 rooms | none | **not done**: needs a LiDAR iPhone in the rooms |
| Fix loop: declaration, root cause + evidence, shipped fix, before/after regenerable, diff | docs/fix_loop_declaration.md, docs/fix_loop_result.md, scripts/fixloop_regen.sh | done (prediction missed; post-mortem) |
| Process evidence: commit history | git log | done |
| README to running on a fresh capture in < 15 min | README.md | done |
| Reproduction bundle | scripts/reproduce.sh, fetch_models.py, requirements-lock.txt, docs/benchmark/*.json | done (fresh clone: LiDAR, drift, synthetic, damage numbers bit-identical; photo/video need the locked versions) |
| Benchmark report: gates at all tiers, repeatability, head-to-head, timing | docs/report.md section 5 | partial (no head-to-head) |
| Technical report <= 6 pages | docs/report.md, docs/report.pdf | done |
| Raw benchmark data | sample captures (supplied) + synthetic generator + manifests | partial (no tape data, no app exports) |
| Weights fetched by script; runs without our infrastructure | scripts/fetch_models.py (or .sh), checksum-verified | done |
| Cover mirrors, glass, wet-look, low light | docs/report.md section 7 | done (described; not measured) |
