# Outputs on the supplied sample data

What `python -m roomplan <capture>` wrote for each sample capture (final code, `requirements-lock.txt`
versions). Each folder holds `plan.json` (schema: `roomplan/schema.json`) and `plan.png`.

| Folder | Input | Command | Result |
|---|---|---|---|
| `room_lidar` | `single_room/c00a170fe1` (37 s) | `python -m roomplan single_room` | 3 rooms, 2 openings, 0 damage regions |
| `floor_only_lidar` | `single_scan_floor_only/1a8384c3f6` (115 s) | `python -m roomplan single_scan_floor_only` | 11 rooms, 15 openings, ceilings `not_observed` (the capture never looked up), 0 damage regions |
| `with_ceiling_lidar` | `single_scan_with_ceiling/c7d28f72c6` (215 s, 100 m walk) | `python -m roomplan single_scan_with_ceiling` | 12 rooms, 13 openings, 11 measured ceilings; damage not run here (see below) |
| `floor_only_photos_photo` | 55 photos in 7 room folders cut from the floor_only video (`scripts/make_photo_set.py`, manifest in `docs/benchmark/`) | `python -m roomplan floor_only_photos --tier photo` | 7 rooms, footprint 0.96 x LiDAR |
| `room_video_video` | the single_room `rgb.mp4` alone (no depth, no poses) | `python -m roomplan room_video` | 4 rooms from the largest registered stretch (43 of 223 frames), footprint 0.58 x LiDAR |
| `staged_damage_room` | single_room with a synthetic stain and crack painted into its frames (`scripts/inject_damage.py`) | see `scripts/reproduce.sh` | crack 0.62 m (truth 0.60), stain 0.141 m^2 (truth 0.114); `wall_z+3.55_elevation.jpg` is the wall image the detector writes for review |

`with_ceiling`: its `rgb.mp4` (252 MB) could not be transferred to our test machine, so damage detection has
not been run on it with real colour (the plan geometry uses depth and poses only). On a copy with `rgb.mp4`
the command runs damage detection too. It was exercised end to end with a stand-in video of the same length
(the capture's depth rendered as grey): 108 frames on 42 walls, no error, 124 s for the whole command. That
checks the stage runs at that size; it says nothing about detection quality on that capture.
