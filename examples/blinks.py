import sys
import time
from tqdm import tqdm

import cv2
import numpy as np

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import pupil_labs.neon_recording as nr # noqa


def make_overlaid_video(recording_dir, output_video_path, fps=30):
    recording = nr.load(recording_dir)

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (recording.eye.width, recording.eye.height),
    )

    output_timestamps = np.arange(
        recording.eye.ts[0], recording.eye.ts[-1], int(1e9 / fps)
    )
    eye_frames = recording.eye.sample(output_timestamps)

    blink_itr = iter(recording.blinks)
    next_blink = next(blink_itr)

    for frame in tqdm(eye_frames):
        frame_pixels = frame.bgr
        while next_blink is not None and next_blink.blink_end_ts_ns < frame.ts:
            try:
                next_blink = next(blink_itr)
            except StopIteration:
                next_blink = None

        in_blink = next_blink is not None and next_blink.blink_start_ts_ns < frame.ts < next_blink.blink_end_ts_ns
        if in_blink:
            frame_pixels = cv2.putText(
                frame_pixels,
                "Blink",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA,
            )

        video_writer.write(frame_pixels)
        cv2.imshow("Frame", frame_pixels)
        cv2.pollKey()

    video_writer.release()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "blink-overlay-output-video.mp4")
