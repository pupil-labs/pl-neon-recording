import sys

import cv2
import numpy as np
import pupil_labs.neon_recording as nr

rec = nr.load(sys.argv[1])

gaze = rec.gaze
eye = rec.eye
scene = rec.scene


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


video = cv2.VideoWriter(
    "video.mp4", cv2.VideoWriter.fourcc("M", "P", "4", "V"), 30, (1600, 1200)
)
try:
    start_ts = rec.unique_events["recording.begin"]
    end_ts = rec.unique_events["recording.end"]

    my_ts = np.arange(start_ts, end_ts, np.mean(np.diff(scene.ts)))

    for gaze_datum, eye_frame, scene_frame in zip(
        gaze.sample(my_ts), eye.sample(my_ts), scene.sample(my_ts)
    ):
        scn_img = (
            scene_frame.cv2
            if scene_frame is not None
            else np.ones((1200, 1600, 3), dtype="uint8") * 128  # gray frames
        )
        ey_img = (
            eye_frame.cv2
            if eye_frame is not None
            else np.zeros((192, 384, 3), dtype="uint8")  # black frames
        )

        overlay_image(scn_img, ey_img, 0, 0)
        if gaze_datum:
            cv2.circle(
                scn_img, (int(gaze_datum.x), int(gaze_datum.y)), 50, (0, 0, 255), 10
            )

        video.write(scn_img)
finally:
    video.release()
