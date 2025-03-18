import sys

import cv2
import numpy as np
from tqdm import tqdm

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import pupil_labs.neon_recording as nr  # noqa: E402, I001
from pupil_labs.neon_recording.stream.av_stream.video_stream import GrayFrame  # noqa: E402


def make_overlaid_video(recording_dir, output_video_path, fps=None):
    recording = nr.open(recording_dir)

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (recording.scene.width, recording.scene.height),
    )

    if fps is None:
        output_timestamps = recording.scene.ts
    else:
        output_timestamps = np.arange(
            recording.scene.ts[0], recording.scene.ts[-1], 1e9 / fps
        )

    scene_datas = recording.scene.sample(output_timestamps)
    combined_data = zip(
        output_timestamps,
        scene_datas,
        recording.gaze.sample(output_timestamps),
    )

    for ts, scene_frame, gaze_datum in tqdm(
        combined_data, total=len(output_timestamps)
    ):
        if abs(scene_frame.ts - ts) < 2e9 / fps:
            frame_pixels = scene_frame.bgr
        else:
            frame_pixels = GrayFrame(scene_frame.width, scene_frame.height).bgr

        if abs(gaze_datum.ts - ts) < 2e9 / fps:
            frame_pixels = cv2.circle(
                frame_pixels,
                (int(gaze_datum.x), int(gaze_datum.y)),
                50,
                (0, 0, 255),
                10,
            )

        video_writer.write(frame_pixels)
        cv2.imshow("Frame", frame_pixels)
        cv2.pollKey()

    video_writer.release()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "gaze-overlay-output-video.avi", 24)
