import sys

import cv2
import numpy as np
import pupil_labs.neon_recording as nr
from pupil_labs.neon_recording.stream.av_stream.video_stream import GrayFrame

from tqdm import tqdm


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


def make_overlaid_video(recording_dir, output_video_path, fps=None):
    recording = nr.load(recording_dir)

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        (recording.scene.width, recording.scene.height)
    )

    if fps is None:
        output_timestamps = recording.scene.ts
    else:
        output_timestamps = np.arange(recording.scene.ts[0], recording.scene.ts[-1], 1 / fps)

    combined_data = zip(
        output_timestamps,
        recording.scene.sample(output_timestamps),
        recording.eye.sample(output_timestamps),
    )

    frame_idx = 0
    for ts, scene_frame, eye_frame in tqdm(combined_data, total=len(output_timestamps)):
        frame_idx += 1
        if abs(scene_frame.ts - ts) < 2 / fps:
            # if the video frame timestamp is too far ahead or behind temporally, replace it with a gray frame
            frame_pixels = scene_frame.bgr
        else:
            frame_pixels = GrayFrame(scene_frame.width, scene_frame.height).bgr

        if abs(eye_frame.ts - ts) < 2 / fps:
            # if the video frame timestamp is too far ahead or behind temporally, replace it with a gray frame
            eye_pixels = cv2.cvtColor(eye_frame.gray, cv2.COLOR_GRAY2BGR)
        else:
            eye_pixels = GrayFrame(eye_frame.width, eye_frame.height).bgr

        overlay_image(frame_pixels, eye_pixels, 50, 50)

        video_writer.write(frame_pixels)
        cv2.imshow('Frame', frame_pixels)
        cv2.pollKey()

    video_writer.release()


if __name__ == '__main__':
    make_overlaid_video(sys.argv[1], "eye-overlay-output-video.avi", 24)
