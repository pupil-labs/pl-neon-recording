import sys

import cv2
import numpy as np

import pupil_labs.neon_recording as nr

if len(sys.argv) < 2:
    print("Usage:")
    print("python basic_usage.py path/to/recording/folder")

# load a recording
recording = nr.load(sys.argv[1])

# get basic info
print("Recording Info:")
print(f"\tStart time (ns): {recording.start_ts_ns}")
print(f"\tWearer         : {recording.wearer['name']}")
print(f"\tDevice serial  : {recording.device_serial}")
print(f"\tGaze samples   : {len(recording.gaze)}")
print("")


# read 10 gaze samples
print("First 10 gaze samples:")
gaze_data = recording.gaze[:10]
for gaze_datum in gaze_data:
    print(f"\t{gaze_datum.ts:0.3f} : ({gaze_datum.x:0.2f}, {gaze_datum.y:0.2f})")
print("")


# resample by interpolation
print("First 10 gaze samples @ 30Hz:")
fps = 30
timestamps = np.arange(
    recording.gaze.ts[0], recording.gaze.ts[-1], 1e9 / fps, dtype=int
)

subsample = recording.gaze.interpolate(timestamps[:10])
for gaze_datum in subsample:
    print(f"\t{gaze_datum.ts:0.3f} : ({gaze_datum.x:0.2f}, {gaze_datum.y:0.2f})")
print("")


# get closest gaze data to scene frame timestamps
matched_gazes = recording.gaze.sample(recording.scene.timestamps)

# interpolate gaze data to scene frame timestamps
interpolated_gazes = recording.gaze.interpolate(recording.scene.timestamps)

# visualize both
scene_gaze_pairs = zip(recording.scene, matched_gazes, interpolated_gazes)
for scene_frame, matched_gaze, interpolated_gaze in scene_gaze_pairs:
    scene_img = scene_frame.bgr
    # draw the nearest-time gaze sample in red
    if matched_gaze is not None:
        scene_img = cv2.circle(
            scene_img, tuple(matched_gaze.xy.astype(int)), 50, (0, 0, 255), 10
        )

    # draw the interpolated gaze sample in blue
    if interpolated_gaze:  # interpolation will fail at the end of the stream
        scene_img = cv2.circle(
            scene_img,
            tuple(interpolated_gaze.xy.astype(int)),
            50,
            (255, 0, 0),
            10,
        )
    else:
        print("Interpolation fail")

    cv2.imshow("Gaze sample comparison", scene_img)
    cv2.pollKey()

cv2.destroyAllWindows()

# Extract data by index
print("64th Gaze Sample:", recording.gaze[64])
cv2.imshow("64th scene frame (push any key to quit)", recording.scene[64].bgr)
cv2.waitKey(0)
