import pprint
import sys

import pupil_labs.neon_recording as nr

rec = nr.load(sys.argv[1])

print("wearer")
pprint.pprint(rec.wearer)
print()

print("recording info:")
pprint.pprint(rec.info)
print()

print("scene camera calibration values:")
pprint.pprint(rec.scene_camera_calibration)
print()

print("right eye camera calibration values:")
pprint.pprint(rec.right_eye_camera_calibration)
print()

print("left eye camera calibration values:")
pprint.pprint(rec.left_eye_camera_calibration)
print()

print("available data streams:")
pprint.pprint(rec.streams)
print()

gaze_data = rec.streams["gaze"].data  # or gaze_data = rec.gaze.data
gaze_ts = rec.streams["gaze"].ts  # or gaze_ts = rec.gaze.ts
gaze_start_offset = rec.streams["gaze"].ts[0] - rec.start_ts
# there is also rec.streams['gaze'].ts_rel

scene = rec.scene

for stream in rec.streams.values():
    if isinstance(stream, nr.stream.av_stream.AudioVideoStream):
        continue

    name = stream.name

    print(name)
    avg_spf = (stream.ts[-1] - stream.ts[0]) / len(stream.ts)
    avg_fps = 1 / avg_spf
    print("avg fps: " + str(avg_fps))
    start_offset = stream.ts[0] - rec.start_ts
    print("start offset: " + str(start_offset))
    print()
