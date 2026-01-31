import sys

import cv2
import numpy as np
from tqdm import tqdm

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import pupil_labs.neon_recording as nr  # noqa: E402
from pupil_labs.video import Writer  # noqa: E402


def make_overlaid_video(recording_dir, output_video_path):
    rec = nr.open(recording_dir)

    combined_data = zip(
        rec.scene,
        rec.gaze.sample(rec.scene.time),
        rec.gaze_monocular_left.sample(rec.scene.time),
        rec.gaze_monocular_right.sample(rec.scene.time),
        strict=True,
    )

    video_writer = Writer(output_video_path)
    video_start_time = rec.scene.time[0]

    gaze_colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
    ]

    for scene_frame, left_gaze, right_gaze, binocular_gaze in tqdm(
        combined_data, total=len(rec.scene.time)
    ):
        # Prepare the next frame
        frame_pixels = scene_frame.bgr

        for gaze_datum, color in zip(
            [binocular_gaze, left_gaze, right_gaze], gaze_colors
        ):
            frame_pixels = cv2.circle(
                frame_pixels,
                (int(gaze_datum.point[0]), int(gaze_datum.point[1])),
                50,
                color,
                10,
            )

        video_time = (scene_frame.time - video_start_time) / 1e9
        video_writer.write_image(frame_pixels, time=video_time)

        # Render current frame and mark start time
        cv2.imshow("Frame", frame_pixels)
        cv2.waitKey(30)

    video_writer.close()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "gaze-overlay-output-video.mp4")
