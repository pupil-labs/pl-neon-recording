import cv2
import numpy as np

import pupil_labs.neon_recording as nr

rec = nr.load('./tests/test_data/2024-01-25_22-19-10_test-f96b6e36/')

gaze = rec.gaze
imus = rec.imu
eye = rec.eye
scene = rec.scene

tstamps = gaze.ts

# nearest neighbor sampling
gz_samps = gaze.sample(tstamps)
imu_samps = imus.sample(tstamps)
eye_samps = eye.sample(tstamps)
scene_samps = scene.sample(tstamps)

c = 0
for gz, imu, ey, scn in zip(gz_samps, imu_samps, eye_samps, scene_samps):
    print('gz', gz)
    print('imu', imu)
    print('eye', eye)
    print('scene', scene)

    c += 1

    if c > 10:
        break


def overlay_image(img, img_overlay, x, y):
    """Overlay `img_overlay` onto `img` at (x, y)."""

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    img_crop[:] = img_overlay_crop


# i-dt fixation detection
dur_thresh = 0.08 # 80 ms in s
disp_thresh = 1.50 # deg

def disp(pts):
    ptsa = np.array(pts)

    # TODO - convert to degrees
    # my head was still in test video, so no need for optic flow just yet
    xs = ptsa['x']
    ys = ptsa['y']

    maxx = np.max(xs)
    minx = np.min(xs)

    maxy = np.max(ys)
    miny = np.min(ys)

    D = (maxx - minx) + (maxy - miny)

    return D


fixs = []
left_idx = 0
while True:
    left_ts = gaze.ts[left_idx]
    if (left_ts + dur_thresh) > gaze.ts[-1]:
        break

    dt = 0.03
    samp = None
    while samp is None:
        samp = gaze.sample_one(left_ts + dur_thresh, dt = dt)
        dt += 0.01

    right_ts = samp.ts
    right_idx = np.where(gaze.ts == right_ts)[0][0]

    btwn_2_ts = gaze.ts[(gaze.ts >= left_ts) & (gaze.ts <= right_ts)]
    pts = [s for s in gaze.sample(btwn_2_ts)]

    if disp(pts) <= disp_thresh:
        while disp(pts) <= disp_thresh:
            right_idx += 1
            pts.append(gaze[right_idx])

        ptsa = np.array(pts)
        cent_x = np.mean(ptsa['x'])
        cent_y = np.mean(ptsa['y'])
        cent_ts = np.mean(ptsa['ts'])

        fixs.append([cent_x, cent_y, cent_ts])

        left_idx = right_idx
    else:
        left_idx += 1

print(np.array(fixs))

video = cv2.VideoWriter('video.mp4', cv2.VideoWriter.fourcc('M','P','4','V'), 30, (1600, 1200))
try:
    evts = rec.unique_events
    start_ts = evts['recording.begin']
    end_ts = evts['recording.end']

    my_ts = np.arange(start_ts, end_ts, np.mean(np.diff(scene.ts)))

    for gz, ey, scn in zip(gaze.sample(my_ts), eye.sample(my_ts), scene.sample(my_ts)):
        scn_img = scn.cv2 if scn is not None else np.ones((1200, 1600, 3), dtype='uint8')*128
        ey_img = ey.cv2 if ey is not None else np.zeros((192, 384, 3), dtype='uint8')

        overlay_image(scn_img, ey_img, 0, 0)
        if gz:
            cv2.circle(scn_img, (int(gz.x), int(gz.y)), 50, (0, 0, 255), 10)

        video.write(scn_img)
finally:
    video.release()
