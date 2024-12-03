import sys

import cv2
import numpy as np
from tqdm import tqdm

import pupil_labs.neon_recording as nr
from pupil_labs.neon_recording.utils import GrayFrame
from pupil_labs.video import Writer


def make_overlaid_video(recording_dir, output_video_path, fps=30):
    recording = nr.load(recording_dir)

    output_timestamps = np.arange(
        recording.scene.abs_timestamps[0],
        recording.scene.abs_timestamps[-1],
        1e9 / fps,
        dtype=int,
    )

    tolerance = int(2e9 / fps)
    matched_data = zip(
        recording.scene.sample(output_timestamps, tolerance=tolerance),
        recording.gaze.sample(output_timestamps, tolerance=tolerance),
    )

    with Writer(output_video_path) as video_writer:
        for scene_frame, gaze_datum in tqdm(matched_data, total=len(output_timestamps)):
            if scene_frame is None:
                scene_frame = GrayFrame(recording.scene.width, recording.scene.height)
            frame_pixels = scene_frame.bgr

            if gaze_datum is not None:
                frame_pixels = cv2.circle(
                    frame_pixels,
                    (int(gaze_datum.x), int(gaze_datum.y)),
                    50,
                    (0, 0, 255),
                    10,
                )

            video_writer.write_image(frame_pixels)
            cv2.imshow("Frame", frame_pixels)
            cv2.pollKey()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "gaze-overlay-output-video.mp4", 24)
