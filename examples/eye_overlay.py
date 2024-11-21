import sys

import cv2
import numpy as np
from tqdm import tqdm

import pupil_labs.neon_recording as nr
from pupil_labs.neon_recording.utils import GrayFrame
from pupil_labs.video import Writer


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


def make_overlaid_video(recording_dir, output_video_path, fps=30):
    recording = nr.load(recording_dir)
    target_timestamps = np.arange(
        recording.eye.timestamps[0],
        recording.scene.timestamps[-1],
        1e9 / fps,
        dtype=int,
    )
    tolerance = int(2e9 / fps)
    matched_data = zip(
        recording.scene.sample(target_timestamps, tolerance=tolerance),
        recording.eye.sample(target_timestamps, tolerance=tolerance),
    )

    with Writer(output_video_path, fps=fps) as video_writer:
        for scene_frame, eye_frame in tqdm(matched_data, total=len(target_timestamps)):
            if scene_frame is None:
                # If no frame exists within the tolerance, replace it with a gray frame
                frame_pixels = GrayFrame(
                    recording.scene.width, recording.scene.height
                ).bgr
            else:
                frame_pixels = scene_frame.bgr

            if eye_frame is None:
                # If no frame exists within the tolerance, replace it with a gray frame
                eye_pixels = GrayFrame(recording.eye.width, recording.eye.height).bgr
            else:
                eye_pixels = eye_frame.bgr

            overlay_image(frame_pixels, eye_pixels, 50, 50)

            video_writer.write_image(frame_pixels)
            cv2.imshow("Frame", frame_pixels)
            cv2.pollKey()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "eye-overlay-output-video.mp4", 24)
