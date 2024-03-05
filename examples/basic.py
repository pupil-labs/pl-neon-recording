import pprint

import pupil_labs.neon_recording as nr

rec = nr.load('./tests/test_data/2024-01-25_22-19-10_test-f96b6e36/')

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


gaze_data = rec.streams['gaze'].data
gaze_ts = rec.streams['gaze'].ts
gaze_start_offset = rec.streams['gaze'].data.ts[0] - rec.start_ts
# there is also rec.streams['gaze'].ts_rel

for stream in rec.streams:
    s = rec.streams[stream]
    name = s.name
    if name == 'scene':
        # not yet implemented
        continue

    print(name)
    avg_spf = (s.data.ts[-1] - s.data.ts[0]) / len(s.data.ts)
    avg_fps = 1 / avg_spf
    print('avg fps: ' + str(avg_fps))
    start_offset = s.data.ts[0] - rec.start_ts
    print('start offset: ' + str(start_offset))
    print()


gaze = rec.streams['gaze']
# rec.gaze is also available

# it might be confusing, but currently, this will only grab two
# timestamps, not three.
# this is probably because i misunderstood the request, but my
# best guess was that we should take successive pairs of ts to be sampled as
# little windows and find the ts that is closest to "current",
# which is the right side of the window
gaze.sample([gaze_ts[0], gaze_ts[1], gaze_ts[2]])

print('iterating over some gaze samples:')
for g in gaze:
    print(g)


print()
print('iterating over some "zip"-ed gaze & imu samples:')
sample_ts = gaze_ts[:15]
for gz, imu in zip(rec.gaze.sample(sample_ts), rec.imu.sample(sample_ts)):
    if gz:
        x = gz.x
        y = gz.y
        ts = gz.ts

        print('gz', x, y, ts)

    if imu:
        pitch = imu.pitch
        yaw = imu.yaw
        roll = imu.roll
        ts = imu.ts

        print('imu', pitch, yaw, roll, ts)


print()
pprint.pprint(rec.unique_events)
print()

event1_ts = rec.unique_events['recording.begin']
event2_ts = rec.unique_events['recording.end']
between_two_events = gaze_ts[(gaze_ts >= event1_ts) & (gaze_ts <= event2_ts)]

print('between the two events:')
print(between_two_events)
print()

gaze_np = gaze.sample(between_two_events).to_numpy()
print('n samples between 2 events: ' + str(len(gaze_np)))
print()

# gets me the closest sample within -+0.01s 
gaze_one_np = gaze.sample_one(event2_ts, dt = 0.01)

# TODO: somehow the recarray features are hidden here until you put
# any of these in a list?
print(gaze_one_np)
print(gaze[-1])
print(gaze[-1] == gaze_one_np)
print()

gaze_one_idx = gaze[42]
print(gaze_one_idx)
print()

print(gaze[42:45])