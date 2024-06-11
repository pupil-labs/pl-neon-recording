import sys

import pupil_labs.neon_recording as nr

rec = nr.load(sys.argv[1])

gaze = rec.gaze
gaze_ts = gaze.ts

# TODO(rob) - we hit end-of-file on last 7 tses, which causes the comprehension to hang?
# all_frames = [f.rgb for f in scene.sample(between_two_events) if f]

# this causes my swap to go crazy and even crashed macos one time
# all_frames = []
# for f in scene.sample(between_two_events):
#     if f:
#         all_frames.append(f.rgb)

# print('all_frames', len(all_frames))
# all_frames_np = np.dstack(all_frames)
# print('all_frames', all_frames_np.shape)

# can this be implemented nicely?
# all_frames_ts = scene.sample(between_two_events).ts
