# Device matrix

| Tier | Phones | Capture app | Input the pipeline reads | Metric scale comes from |
|---|---|---|---|---|
| **LiDAR** | iPhone 12 Pro / Pro Max and every later **Pro** / Pro Max (13, 14, 15, 16, 17 Pro); iPad Pro 2020 or later | Stray Scanner (free) | 256x192 depth at ~46 fps, ARKit poses, per-frame intrinsics, RGB video, IMU | the LiDAR sensor |
| **Video** | any iPhone 15 or newer (15, 15 Plus, 16, 16e, 17, Air and all Pro models); works on older iPhones too | built-in Camera, Video, 1x, landscape | the clip only (no depth, no poses) | camera height prior 1.40 +- 0.07 m |
| **Photo** | any iPhone 15 or newer | built-in Camera, Photo, 1x, portrait | 4-8 stills per room folder (EXIF focal length used) | camera height prior 1.40 +- 0.07 m |

Non-Pro iPhones have no LiDAR: they can only use the video and photo tiers.

Processing machine: any laptop, CPU only, 8 GB RAM, Python 3.10-3.13 and ffmpeg. No GPU, no network after
`scripts/fetch_models.sh`.

## What each tier honestly delivers (sample data; no tape ground truth was available)

Numbers are from `scripts/benchmark.py` on the three sample captures. "Repeatability" compares two
independent LiDAR captures of the same apartment; "vs LiDAR" compares a tier with the LiDAR tier of the same
capture. Neither is accuracy against a tape; see docs/report.md section "Error budget".

| | LiDAR | Video | Photo |
|---|---|---|---|
| wall position, same surface, capture vs capture | ~1 cm (12 of 27 matched planes within 2.3 cm, median 0.5 cm) | n/a | n/a |
| wall position, all matched planes, capture vs capture | median 2.7 cm | n/a | n/a |
| drift on a 100 m, 215 s walk (loop residual, on / off) | 0.0 cm / 11.6 cm | | |
| reported 95% interval, wall length | +-2-4 cm (measured ends), +-20 cm (end bounded by furniture) | +-10-15% | +-10-20% |
| ceiling height | measured where the capture looked up (coverage reported per room); "not observed" otherwise | wide (+-10%) | wide (+-10%) |
| expected scale error before any other error | ~0.5% | 5% (1 sigma) from the height prior | 5% (1 sigma) from the height prior |
| stitched whole-property plan | yes | see report | see report |

Video and photo rows are filled from the latest benchmark in docs/report.md.

## Runtime (2-core cloud VM, CPU)

| capture | tier | runtime |
|---|---|---|
| single_room (37 s, 4 rooms) | LiDAR | ~10-15 s |
| single_scan_floor_only (115 s) | LiDAR | ~30-180 s (with damage frames) |
| single_scan_with_ceiling (215 s, 100 m walk) | LiDAR | ~45-75 s |
