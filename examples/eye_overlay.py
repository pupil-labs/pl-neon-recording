import sys

import cv2
import numpy as np
from tqdm import tqdm

import pupil_labs.neon_recording as nr
from pupil_labs.matching import Matcher
from pupil_labs.video import Writer


class GrayFrame:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self._bgr = None
        self._gray = None

    @property
    def bgr(self):
        if self._bgr is None:
            self._bgr = 128 * np.ones([self.height, self.width, 3], dtype="uint8")

        return self._bgr

    @property
    def gray(self):
        if self._gray is None:
            self._gray = 128 * np.ones([self.height, self.width], dtype="uint8")

        return self._gray


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
        recording.eye.timestamps[0], recording.scene.timestamps[-1], 1 / fps
    )
    matched_data = Matcher(
        target_timestamps,
        [recording.scene, recording.eye],
        tolerance=2 / fps,
    )

    with Writer(output_video_path) as video_writer:
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

            video_writer.write(frame_pixels)
            cv2.imshow("Frame", frame_pixels)
            cv2.pollKey()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "eye-overlay-output-video.mp4", 24)
