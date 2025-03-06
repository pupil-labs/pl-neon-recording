import sys

import cv2
import numpy as np
from tqdm import tqdm

import pupil_labs.neon_recording as nr
from pupil_labs.neon_recording.stream.av_stream.video_stream import GrayFrame


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

        points = [[*v] for v in zip(x_values, y_values)]

        cv2.line(img, points[0], points[1], color, line_width)


def make_eye_state_video(recording_dir, output_video_path):
    recording = nr.load(recording_dir)

    fps = 200

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (recording.eye.width, recording.eye.height),
    )

    output_timestamps = np.arange(
        recording.eye.ts[0], recording.eye.ts[-1], int(1e9 / fps)
    )

    eye_video_sampled = recording.eye.sample(output_timestamps)
    eye_state_sampled = recording.eye_state.sample(output_timestamps)
    combined_data = zip(
        output_timestamps,
        eye_video_sampled,
        eye_state_sampled,
    )

    plot_metas = {
        "optical_axis_left_x": {"color": [0, 0, 255]},
        "optical_axis_left_y": {
            "color": [0, 255, 0],
        },
        "optical_axis_left_z": {
            "color": [255, 0, 0],
        },
    }

    for plot_name, plot_meta in plot_metas.items():
        plot_meta["range"] = (
            np.min(recording.eye_state.data[plot_name]),
            np.max(recording.eye_state.data[plot_name]),
        )

    plot_duration = 0.5
    plot_point_count = plot_duration * fps
    plot_x_width = recording.eye.width / plot_point_count

    for ts, eye_frame, eye_state in tqdm(combined_data, total=len(output_timestamps)):
        if abs(eye_frame.ts - ts) < 2e9 / fps:
            eye_pixels = cv2.cvtColor(eye_frame.gray, cv2.COLOR_GRAY2BGR)
        else:
            eye_pixels = GrayFrame(eye_frame.width, eye_frame.height).bgr

        for plot_name, plot_meta in plot_metas.items():
            min_ts = ts - plot_duration
            time_frame = (min_ts < eye_state_sampled.data.ts) & (
                eye_state_sampled.data.ts <= ts
            )
            plot_data = eye_state_sampled.data[time_frame][plot_name]
            plot(
                eye_pixels,
                plot_data,
                plot_meta["range"],
                plot_x_width,
                plot_meta["color"],
            )

        video_writer.write(eye_pixels)
        cv2.imshow("Frame", eye_pixels)
        cv2.pollKey()

    video_writer.release()


if __name__ == "__main__":
    make_eye_state_video(sys.argv[1], "eye-state-output-video.avi")
