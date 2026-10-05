# Device matrix

| Tier | Phones | Capture app | Input the pipeline reads | Metric scale comes from |
|---|---|---|---|---|
| **LiDAR** | iPhone 12 Pro / Pro Max and every later **Pro** / Pro Max (13, 14, 15, 16, 17 Pro); iPad Pro 2020 or later | Stray Scanner (free) | 256x192 depth at ~46 fps, ARKit poses, per-frame intrinsics, RGB video, IMU | the LiDAR sensor |
| **Video** | any iPhone 15 or newer (15, 15 Plus, 16, 16e, 17, Air and all Pro models); works on older iPhones too | built-in Camera, Video, 1x, landscape | the clip only (no depth, no poses) | camera height prior 1.40 +- 0.07 m |
| **Photo** | any iPhone 15 or newer | built-in Camera, Photo, 1x, portrait | 4-8 stills per room folder (EXIF focal length used) | camera height prior 1.40 +- 0.07 m |

Non-Pro iPhones have no LiDAR: they can only use the video and photo tiers.

Processing machine: any laptop, CPU only, 8 GB RAM, Python 3.10-3.13 and ffmpeg. No GPU, no network after
`scripts/fetch_models.py`.

## What each tier honestly delivers (sample data; no tape ground truth was available)

Numbers are from `scripts/benchmark.py` on the three sample captures. "Repeatability" compares two
independent LiDAR captures of the same apartment; "vs LiDAR" compares a tier with the LiDAR tier of the same
capture. Neither is accuracy against a tape; see docs/report.md section "Error budget".

| | LiDAR | Video | Photo |
|---|---|---|---|
| wall plane position, capture vs capture (30 matched planes) | 8 within 1 cm, 10 within 2 cm, median 2.7 cm | n/a | n/a |
| door width, capture vs capture (7 matched doors) | 2 within 2 cm, median 2.2 cm | n/a | n/a |
| drift on a 100 m, 215 s walk (loop residual, on / off) | 2.1 cm / 11.4 cm | n/a | n/a |
| reported 95% interval, wall length | +-2-4 cm (measured ends), +-20 cm (end bounded by furniture) | +-0.21-0.30 m | +-0.20-0.30 m |
| ceiling height | measured where the capture looked up (coverage reported per room); "not observed" otherwise | wide (+-10%) | wide (+-10%) |
| expected scale error before any other error | ~0.5% | 5% (1 sigma) from the height prior | 5% (1 sigma) from the height prior |
| synthetic ground truth (exact) | walls 24/24 within 2 cm (median 0.8 cm), ceilings within 2 mm, doors 5/6 within 2 cm | not measured | not measured |
| footprint vs LiDAR, same capture | reference | 0.58 (partial clip) | 0.96 |
| too little input (2 photos of a room, a 5 s clip) | n/a | single-view room estimate, 30% sigma | single-view room estimate, 30% sigma |
| stitched whole-property plan | yes | partial (largest registered stretch) | yes, rooms under-segmented (7 vs 11) |


## Runtime (2-core cloud VM, CPU)

| capture | tier | runtime |
|---|---|---|
| single_room (37 s) | LiDAR | 16 s (with damage) |
| single_scan_floor_only (115 s) | LiDAR | 60 s (with damage) |
| single_scan_with_ceiling (215 s, 100 m walk) | LiDAR | 75 s without damage; 124 s with (stand-in video of the same length) |
| floor_only photo folders (55 photos) | photo | ~7-11 min |
| single_room clip (37 s, 6 fps) | video | ~14-18 min |
