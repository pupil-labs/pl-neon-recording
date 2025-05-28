import sys

import cv2
import numpy as np
from tqdm import tqdm

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

from pupil_labs import neon_recording as nr  # noqa: E402


def write_text(image, text, x, y):
    return cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def make_overlaid_video(recording_dir, output_video_path, fps=30):
    recording = nr.open(recording_dir)

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (recording.eye.width, recording.eye.height),
    )

    output_timestamps = np.arange(
        recording.eye.time[0], recording.eye.time[-1], int(1e9 / fps)
    )
    eyes_and_worn = zip(
        recording.eye.sample(output_timestamps),
        recording.worn.sample(output_timestamps),
        strict=False,
    )

    for frame, worn_record in tqdm(eyes_and_worn, total=len(output_timestamps)):
        frame_pixels = frame.bgr

        text_y = 40
        if worn_record.worn:
            frame_pixels = write_text(frame_pixels, "Worn", 0, text_y)

        video_writer.write(frame_pixels)
        cv2.imshow("Frame", frame_pixels)
        cv2.pollKey()

    video_writer.release()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "worn.mp4")
