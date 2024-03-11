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



video = cv2.VideoWriter('video.mp4', cv2.VideoWriter.fourcc('M','P','4','V'), 30, (1600, 1200))
try:
    tstamps = scene.ts
    for gz, ey, scn in zip(gaze.sample(tstamps), eye.sample(tstamps), scene.sample(tstamps)):
        img = scn.cv2

        overlay_image(img, ey.cv2, 0, 0)
        cv2.circle(img, (int(gz.x), int(gz.y)), 50, (0, 0, 255), 10)

        video.write(img)
finally:
    video.release()
