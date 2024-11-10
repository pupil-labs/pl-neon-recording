import sys

import cv2
import numpy as np
from tqdm import tqdm
from utils import GrayFrame

import pupil_labs.neon_recording as nr
from pupil_labs.matching import Matcher
from pupil_labs.video import Writer


def make_overlaid_video(recording_dir, output_video_path, fps=30):
    recording = nr.load(recording_dir)

    output_timestamps = np.arange(
        recording.scene.timestamps[0], recording.scene.timestamps[-1], 1 / fps
    )

    matched_data = Matcher(
        output_timestamps, [recording.scene, recording.gaze], tolerance=2 / fps
    )

    with Writer(output_video_path) as video_writer:
        for scene_frame, gaze_datum in tqdm(matched_data, total=len(output_timestamps)):
            if scene_frame is None:
                frame_pixels = GrayFrame(
                    recording.scene.width, recording.scene.height
                ).bgr
            else:
                frame_pixels = scene_frame.bgr

            if gaze_datum is not None:
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


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "gaze-overlay-output-video.mp4", 24)
