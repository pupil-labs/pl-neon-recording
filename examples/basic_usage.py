import sys

import pupil_labs.neon_recording as nr

if len(sys.argv) < 2:
    print("Usage:")
    print("python basic_usage.py path/to/recording/folder")

# Open a recording
recording = nr.open(sys.argv[1])

# get basic info
print("Recording Info:")
print(f"\tStart time (ns): {recording.start_time}")
print(f"\tWearer         : {recording.wearer['name']}")
print(f"\tDevice serial  : {recording.device_serial}")
print(f"\tGaze samples   : {len(recording.gaze)}")
print("")

# read 10 gaze samples
print("First 10 gaze samples:")
timestamps = recording.gaze.time[:10]
subsample = recording.gaze.sample(timestamps)
for gaze_datum in subsample:
    print(
        f"\t{gaze_datum.time} :",
        f"({gaze_datum.point[0]:0.2f}, {gaze_datum.point[1]:0.2f})"
    )
