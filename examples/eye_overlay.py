import sys
from fractions import Fraction

import cv2
import numpy as np
import pupil_labs.neon_recording as nr
import pupil_labs.video as plv

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

def make_overlaid_video(recording_dir, output_video_path, fps=30):
    recording = nr.load(recording_dir)

    output_container = plv.open(output_video_path, mode="w")
    out_video_stream = output_container.add_stream("libx264", rate=fps)
    out_video_stream.time_base = Fraction(1, fps)

    output_timestamps = np.arange(recording.scene.ts[0], recording.scene.ts[-1], 1/fps)

    combined_data = zip(
        recording.scene.sample(output_timestamps, epsilon=1/15),
        recording.eye.sample(output_timestamps, epsilon=1/15),
    )

    frame_idx = 0
    for scene_frame, eye_frame in tqdm(combined_data, total=len(output_timestamps)):
        frame_idx += 1
        frame_pixels = scene_frame.bgr

        if eye_frame is not None:
            eye_pixels = cv2.cvtColor(eye_frame.gray, cv2.COLOR_GRAY2BGR)
            overlay_image(frame_pixels, eye_pixels, 50, 50)

        output_frame = plv.VideoFrame.from_ndarray(frame_pixels, format='bgr24')
        output_frame.pts = frame_idx

        for packet in out_video_stream.encode(output_frame):
            output_container.mux(packet)

    for packet in out_video_stream.encode():
        output_container.mux(packet)

    output_container.close()


if __name__ == '__main__':
    make_overlaid_video(sys.argv[1], "eye-overlay-output-video.mp4", 24)
