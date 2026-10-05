# Capture protocol (Route 2: stock apps). One page; follow it literally.

Pick ONE of the three ways to capture. Every way ends with "one folder per capture", and the computer
runs one command on that folder: `python -m roomplan <folder>`.

## Before you start (all three ways)
- Turn on **all the lights** and open the curtains. Close the doors of any **mirrored** wardrobes.
- Walk **slowly** the whole time (about one step per second) and hold the phone at **chest height**.
- Never cover the camera with your fingers. Do not change the zoom while capturing.

## A. LiDAR scan (iPhone 12 Pro or newer Pro/Pro Max, or iPad Pro): most accurate
1. Install **Stray Scanner** (App Store, free, by Stray Robots). Open it and allow camera access.
2. Stand in the room by the front door. Press the **red record button**.
3. Walk **through every room** in the property. In each room:
   a. Walk around the room **along the walls**, about 1 m away from them, pointing the phone at the
      walls (not at the floor).
   b. Once per room, **slowly tilt the phone up to the ceiling and back down** (2 seconds up, 2 down).
   c. Point the phone at every **door frame and window** for a moment as you pass it.
4. **Finish where you started**, by the front door, pointing at the same wall you started on. This
   lets the software check and correct drift.
5. Press the record button again to stop. Total time: about 30 s per room (a 4-room flat takes ~2-3 min).
6. Transfer: in Stray Scanner tap the scan → **Export** → save to Files → AirDrop/copy the whole folder
   (it contains `rgb.mp4`, `depth/`, `odometry.csv`, ...) to the computer. Do not rename files inside.

## B. Video (any iPhone 15 or newer)
1. Open the built-in **Camera** app → **Video**. Use the **1×** lens. Hold the phone **sideways
   (landscape)**.
2. Start at the front door and press record. Walk through every room as in A3 (along the walls, tilt up to
   the ceiling once per room, look at each door frame). **Keep some floor in view most of the time**
   (point the phone slightly downwards, about 20°).
3. **Turn slowly** (count to three for a quarter turn). Fast turns break the reconstruction.
4. Finish where you started, looking at the same wall. Stop recording. 1-4 minutes is right.
5. Transfer: AirDrop the video to the computer and put it alone in a new folder.

## C. Photos (any iPhone 15 or newer)
1. Open **Camera** → **Photo**, **1×** lens, phone **upright** (portrait). Leave all settings default.
2. Make one folder per room on the computer later; name it after the room (`kitchen`, `bedroom1`, ...).
3. In each room take **4 to 8 photos** while walking slowly around it:
   - take a photo **every half step or every time you have turned about 30°** (one-third of a quarter
     turn), so each photo **shows about half of what the previous one showed**;
   - **every photo must show some floor** at the bottom, and the wall in front of you;
   - stand about 1 m from the walls; do not photograph a blank wall from close up.
4. **Doorway photo:** when you walk into the next room, take the first photo of that room **standing in the
   doorway**, looking into it, and the second one turned back towards the room you just left. These two
   shots are what lets the software join the rooms together. Put them in the **new** room's folder.
5. Transfer: AirDrop all photos; on the computer make one sub-folder per room inside one capture folder:
   `capture/kitchen/IMG_0001.HEIC ...`, `capture/bedroom1/...`. HEIC and JPEG both work.

## Things that go wrong (and what to do)
| Problem | Do this |
|---|---|
| Mirror, glass shower screen, glossy black TV | Capture it, but do not stand still facing it; walk past at an angle |
| Very dark room | Turn lights on; if impossible, use LiDAR (A) |
| Someone walks in front of the camera | Fine; keep going |
| Phone gets hot / app stops | Save what you have, start a new capture from that room, include one room already captured |

## What you get
`plan.json` (every measurement with a 95% interval) and `plan.png` (the stitched floor plan). Photos and video
give wider intervals than LiDAR because their size comes from the phone height (see device matrix).
