import sys

import cv2
import numpy as np
from tqdm import tqdm

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import pupil_labs.neon_recording as nr  # noqa: E402
from pupil_labs.video import Writer  # noqa: E402


def make_overlaid_video(recording_dir, output_video_path, fps=None):
    # Open a recording
    recording = nr.open(recording_dir)

    video_writer = Writer(output_video_path)
    video_start_time = recording.scene.time[0]

    # get closest gaze data to scene frame timestamps
    matched_gazes = recording.gaze.sample(recording.scene.time)

    # interpolate gaze data to scene frame timestamps
    interpolated_gazes = recording.gaze.interpolate(recording.scene.time)

    # visualize both
    scene_gaze_pairs = zip(
        recording.scene, matched_gazes, interpolated_gazes, strict=False
    )
    for scene_frame, matched_gaze, interpolated_gaze in tqdm(
        scene_gaze_pairs, total=len(recording.scene)
    ):
        # draw the nearest-time gaze sample in red
        frame = cv2.circle(
            scene_frame.bgr,
            (int(matched_gaze.point[0]), int(matched_gaze.point[1])),
            50,
            (0, 0, 255),
            10,
        )

        # draw the interpolated gaze sample in blue
        if not np.isnan(interpolated_gaze.point[0]) and not np.isnan(
            interpolated_gaze.point[1]
        ):
            frame = cv2.circle(
                frame,
                (int(interpolated_gaze.point[0]), int(interpolated_gaze.point[1])),
                50,
                (255, 0, 0),
                10,
            )

        video_time = (scene_frame.time - video_start_time) / 1e9
        video_writer.write_image(frame, time=video_time)
        cv2.imshow("Gaze sample comparison", frame)
        cv2.pollKey()

    cv2.destroyAllWindows()
    video_writer.close()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "interpolation-compare.mp4", 30)
