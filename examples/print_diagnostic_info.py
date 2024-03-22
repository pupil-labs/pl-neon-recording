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

print("scene camera info:")
pprint.pprint(rec.scene_camera)
print()

print("eye 1 camera info:")
pprint.pprint(rec.eye1_camera)
print()

print("eye 2 camera info:")
pprint.pprint(rec.eye2_camera)
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
    name = stream.name

    print(name)
    avg_spf = (stream.ts[-1] - stream.ts[0]) / len(stream.ts)
    avg_fps = 1 / avg_spf
    print("avg fps: " + str(avg_fps))
    start_offset = stream.ts[0] - rec.start_ts
    print("start offset: " + str(start_offset))
    print()
