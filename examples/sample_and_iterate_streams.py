import sys

import pupil_labs.neon_recording as nr

rec = nr.load(sys.argv[1])

print()
print("iterating over first 15 nearest neighbor samples from all streams:")
sample_ts = rec.gaze.ts[:15]

combined_data = zip(
    rec.gaze.sample(sample_ts),
    rec.imu.sample(sample_ts),
    rec.scene.video_stream.sample(sample_ts),
    rec.eye.sample(sample_ts),
)
for gaze_datum, imu_datum, scene_frame, eye_frame in combined_data:
    if gaze_datum:
        x = gaze_datum.x
        y = gaze_datum.y
        ts = gaze_datum.ts

        print("gaze_datum", x, y, ts)

    if imu_datum:
        pitch = imu_datum.pitch
        yaw = imu_datum.yaw
        roll = imu_datum.roll
        ts = imu_datum.ts

        print("imu_datum", pitch, yaw, roll, ts)

    if scene_frame:
        img = scene_frame.rgb
        gray = scene_frame.gray
        img_index = scene_frame.index  # frame index in the stream
        img_ts = scene_frame.ts
        time_into_the_scene_stream = img_ts - rec.start_ts
        print("scene_frame", img_ts)

    if eye_frame:
        img = eye_frame.gray
        print("eye_frame", img.shape)


gaze = rec.gaze

# gets me the closest sample within -+0.01s
gaze_single_sample = gaze.sample_one(gaze.ts[-20], dt=0.01)

print()
print(gaze_single_sample)
print()

gaze_single_indexed = gaze[42]
print(gaze_single_indexed)
print()

print(gaze[42:45])
print()

# get all samples in a list
gaze_samples_list = list(gaze.sample(gaze.ts[:15]))

# get all samples as a numpy recarray (gaze/imu) or ndarray of frames (video)
gaze_samples_np = nr.sampled_to_numpy(gaze.sample(gaze.ts[:15]))

# NOTE: the following is quite intense on the RAM.
scene_samples_np = nr.sampled_to_numpy(
    rec.scene.video_stream.sample(rec.scene.video_stream.ts[:15])
)
print(scene_samples_np.shape)
