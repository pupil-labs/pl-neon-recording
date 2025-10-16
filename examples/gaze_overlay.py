import sys

import cv2
import numpy as np
from tqdm import tqdm

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

from pupil_labs.video import Writer  # noqa: E402

import pupil_labs.neon_recording as nr  # noqa: E402


def make_overlaid_video(recording_dir, output_video_path):
    rec = nr.open(recording_dir)

    combined_data = zip(
        rec.scene,
        rec.gaze.sample(rec.scene.time),
        strict=True,
    )

    video_writer = Writer(output_video_path)
    video_start_time = rec.scene.time[0]

    for scene_frame, gaze_datum in tqdm(combined_data, total=len(rec.scene.time)):
        frame_pixels = scene_frame.bgr

        frame_pixels = cv2.circle(
            frame_pixels,
            (int(gaze_datum.point[0]), int(gaze_datum.point[1])),
            50,
            (0, 0, 255),
            10,
        )

        video_time = (scene_frame.time - video_start_time) / 1e9
        video_writer.write_image(frame_pixels, time=video_time)

        cv2.imshow("Frame", frame_pixels)
        cv2.waitKey(30)

    video_writer.close()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "gaze-overlay-output-video.mp4")
