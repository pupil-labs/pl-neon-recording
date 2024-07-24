import sys

import numpy as np
import cv2

import pupil_labs.neon_recording as nr
from pupil_labs.neon_recording.stream import InterpolationMethod

if len(sys.argv) < 2:
    print("Usage:")
    print("python basic_usage.py path/to/recording/folder")

# load a recording
recording = nr.load(sys.argv[1])


# get basic info
print("Recording Info:")
print(f"\tStart time (ns): {recording.start_ts_ns}")
print(f"\tWearer         : {recording.wearer['name']}")
print(f"\tDevice serial  : {recording.serial}")
print("")


# read 10 gaze samples
print("First 10 gaze samples:")
timestamps = recording.gaze.ts[:10]
subsample = recording.gaze.sample(timestamps)
for gaze_datum in subsample:
    print(f"\t{gaze_datum.ts:0.3f} : ({gaze_datum.x:0.2f}, {gaze_datum.y:0.2f})")
print("")


# resample by interpolation
print("First 10 gaze samples @ 30Hz:")
fps = 30
timestamps = np.arange(
    recording.gaze.ts[0],
    recording.gaze.ts[-1],
    1 / fps
)

subsample = recording.gaze.sample(timestamps[:10], InterpolationMethod.LINEAR)
for gaze_datum in subsample:
    print(f"\t{gaze_datum.ts:0.3f} : ({gaze_datum.x:0.2f}, {gaze_datum.y:0.2f})")
print("")


# get closest gaze data to scene frame timestamps
matched_gazes = recording.gaze.sample(recording.scene.ts)

# interpolate gaze data to scene frame timestamps
interpolated_gazes = recording.gaze.sample(recording.scene.ts, InterpolationMethod.LINEAR)

# visualize both
scene_gaze_pairs = zip(recording.scene, matched_gazes, interpolated_gazes)
for scene_frame, matched_gaze, interpolated_gaze in scene_gaze_pairs:
    if matched_gaze:
        frame = cv2.circle(
            scene_frame.bgr,
            (int(matched_gaze.x), int(matched_gaze.y)),
            50, (0, 0, 255), 10
        )

    if interpolated_gaze:
        frame = cv2.circle(
            frame,
            (int(interpolated_gaze.x), int(interpolated_gaze.y)),
            50, (255, 0, 0), 10
        )

    cv2.imshow('Gaze sample comparison', frame)
    cv2.pollKey()
