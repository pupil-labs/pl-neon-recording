import sys

import cv2
import numpy as np

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import pupil_labs.neon_recording as nr  # noqa: E402
from pupil_labs.video import Writer  # noqa: E402


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


def plot(img, data, value_range, x_width, color, line_width=2):
    for idx in range(1, len(data)):
        x_values = [int(idx2 * x_width) for idx2 in [idx - 1, idx]]

        y_norms = [
            (data[idx2] - value_range[0]) / (value_range[1] - value_range[0])
            for idx2 in [idx - 1, idx]
        ]
        y_values = [int(y_norm * img.shape[0]) for y_norm in y_norms]

        points = [[*v] for v in zip(x_values, y_values, strict=False)]

        cv2.line(img, points[0], points[1], color, line_width)


def make_eye_state_video(recording_dir, output_video_path):
    recording = nr.open(recording_dir)

    fps = 200
    video_writer = Writer(output_video_path)
    video_start_time = recording.eye.time[0]

    plot_config = [
        {"color": [0, 0, 255]},
        {"color": [0, 255, 0]},
        {"color": [255, 0, 0]},
    ]

    for dim, config in enumerate(plot_config):
        config["range"] = (
            np.min(recording.eyeball_pose.optical_axis_left[:, dim]),
            np.max(recording.eyeball_pose.optical_axis_left[:, dim]),
        )

    plot_duration_secs = 0.5
    plot_point_count = plot_duration_secs * fps
    plot_x_width = recording.eye.width / plot_point_count

    for eye_sample in recording.eye:
        eye_pixels = eye_sample.bgr

        for dim, config in enumerate(plot_config):
            min_ts = eye_sample.time - plot_duration_secs * 1e9
            mask = (min_ts < recording.eyeball_pose.time) & (
                recording.eyeball_pose.time <= eye_sample.time
            )
            plot_data = recording.eyeball_pose.optical_axis_left[mask, dim]
            plot(
                eye_pixels,
                plot_data,
                config["range"],
                plot_x_width,
                config["color"],
            )

        video_time = (eye_sample.time - video_start_time) / 1e9
        video_writer.write_image(eye_pixels, time=video_time)
        cv2.imshow("Frame", eye_pixels)
        cv2.pollKey()

    video_writer.close()


if __name__ == "__main__":
    make_eye_state_video(sys.argv[1], "eye-state-output-video.avi")
