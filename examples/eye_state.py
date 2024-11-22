import sys

import cv2
import numpy as np
from tqdm import tqdm

import pupil_labs.neon_recording as nr
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


def plot(img, data, value_range, color, line_width=2):
    if len(data) < 2:
        return

    x_width = img.shape[1] / len(data)
    for idx in range(1, len(data)):
        x_values = [int(idx2 * x_width) for idx2 in [idx - 1, idx]]

        y_norms = [
            (data[idx2] - value_range[0]) / (value_range[1] - value_range[0])
            for idx2 in [idx - 1, idx]
        ]
        y_values = [int(y_norm * img.shape[0]) for y_norm in y_norms]

        points = [[*v] for v in zip(x_values, y_values)]

        cv2.line(img, points[0], points[1], color, line_width)


def make_eye_state_video(recording_dir, output_video_path):
    recording = nr.load(recording_dir)

    fps = 200
    plot_metas = {
        0: {"color": [0, 0, 255]},
        1: {
            "color": [0, 255, 0],
        },
        2: {
            "color": [255, 0, 0],
        },
    }

    for dim, plot_meta in plot_metas.items():
        plot_meta["range"] = (
            np.min(recording.eye_state.optical_axis_left[dim]),
            np.max(recording.eye_state.optical_axis_left[dim]),
        )

    plot_duration = int(1e9 * 0.5)

    with Writer(output_video_path, fps=fps) as video_writer:
        for eye_frame in tqdm(recording.eye):
            eye_pixels = eye_frame.bgr

            for dim, plot_meta in plot_metas.items():
                min_ts = eye_frame.timestamp - plot_duration
                time_frame = (min_ts < recording.eye_state.timestamps) & (
                    recording.eye_state.timestamps <= eye_frame.timestamp
                )
                plot_data = recording.eye_state.optical_axis_left[time_frame, dim]
                plot(
                    eye_pixels,
                    plot_data,
                    plot_meta["range"],
                    plot_meta["color"],
                )

            video_writer.write_image(eye_pixels)

            cv2.imshow("Frame", eye_pixels)
            cv2.pollKey()


if __name__ == "__main__":
    make_eye_state_video(sys.argv[1], "eye-state-output-video.avi")
