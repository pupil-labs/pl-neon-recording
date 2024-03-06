import pprint

import cv2
import numpy as np
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
gaze_start_offset = rec.streams['gaze'].ts[0] - rec.start_ts
# there is also rec.streams['gaze'].ts_rel

scene = rec.streams['scene']
print(scene.ts)
print()

for stream in rec.streams:
    s = rec.streams[stream]
    name = s.name

    print(name)
    avg_spf = (s.ts[-1] - s.ts[0]) / len(s.ts)
    avg_fps = 1 / avg_spf
    print('avg fps: ' + str(avg_fps))
    start_offset = s.ts[0] - rec.start_ts
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
gaze.sample_rob([gaze_ts[0], gaze_ts[1], gaze_ts[2]])

print('iterating over some gaze samples:')
for g in gaze:
    print(g)


print()
print('iterating over some "zip"-ed gaze & imu samples:')
sample_ts = gaze_ts[:15]
for gz, imu, scene_frame, eye in zip(rec.gaze.sample_rob(sample_ts), rec.imu.sample_rob(sample_ts), rec.scene.sample_rob(sample_ts), rec.eye.sample_rob(sample_ts)):
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

    if scene_frame:
        img = scene_frame.rgb
        gray = scene_frame.gray
        img_index = scene_frame.index # frame index in the stream
        # img_ts = scene_frame.ts # TODO(rob) - fix this: same as rec.streams['scene'].ts[world.index]
        img_ts = rec.streams['scene'].ts[scene_frame.index]
        time_into_the_stream = img_ts - rec.start_ts
        print('scene', img_ts)

    if eye:
        img = eye.gray

        print('eye', img.shape)


print()
pprint.pprint(rec.unique_events)
print()

event1_ts = rec.unique_events['recording.begin']
event2_ts = rec.unique_events['recording.end']
between_two_events = gaze_ts[(gaze_ts >= event1_ts) & (gaze_ts <= event2_ts)]

print('between the two events:')
print(between_two_events)
print()

gaze_np = gaze.sample_rob(between_two_events).to_numpy()
print('n samples between 2 events: ' + str(len(gaze_np)))
print()


# TODO(rob) - we hit end-of-file on last 7 tses, which causes the comprehension to hang?
# all_frames = [f.rgb for f in scene.sample(between_two_events) if f]
all_frames = []
for f in scene.sample_rob(between_two_events):
    if f:
        all_frames.append(f.rgb)


print('all_frames', len(all_frames))
# all_frames_np = np.dstack(all_frames)
# print('all_frames', all_frames_np.shape)
        
# need to implement nicely
# all_frames_ts = scene.sample(between_two_events).ts 

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


scene_video = rec.scene
gaze = rec.gaze

between_two_events = scene_video.ts[(scene_video.ts >= event1_ts) & (scene_video.ts <= event2_ts)]

video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1600, 1200))

try:
    for gaze, world in zip(gaze.sample_rob(between_two_events), scene_video.sample_rob(between_two_events)):
        img = world.cv2

        if gaze:
            cv2.circle(img, (int(gaze.x), int(gaze.y)), 50, (0, 0, 255), 10)
            # cv2.imshow('gaze', img)
            # cv2.imwrite('gaze.png', img)
            video.write(img)
except:
    video.release()
    raise
