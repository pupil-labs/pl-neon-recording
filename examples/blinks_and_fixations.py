import sys

import cv2
import numpy as np
from tqdm import tqdm

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

from pupil_labs.video import Writer  # noqa: E402

import pupil_labs.neon_recording as nr  # noqa: E402
from pupil_labs.neon_recording import match_ts  # noqa: E402


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


def match_events(target_time, events):
    # Blink start needs to be <= target_time
    matches = match_ts(target_time, events.start_time, method="backward")

    # Blink end needs to be >= target_time
    matches_end = match_ts(target_time, events.stop_time, method="forward")

    matches[np.isnan(matches_end)] = np.nan
    matches[matches != matches_end] = np.nan

    return matches


def make_overlaid_video(recording_dir, output_video_path):
    recording = nr.open(recording_dir)

    video_writer = Writer(
        output_video_path,
    )
    video_start_time = recording.eye.time[0]

    blink_matches = match_events(recording.eye.time, recording.blinks)
    fixation_matches = match_events(recording.eye.time, recording.fixations)

    blink_count = 0
    fixation_count = 0
    for frame, blink_index, fixation_index in tqdm(
        zip(recording.eye, blink_matches, fixation_matches, strict=False),
        total=len(recording.eye)
    ):
        frame_pixels = frame.bgr

        if not np.isnan(blink_index):
            blink_count = int(blink_index + 1)
        frame_pixels = write_text(frame_pixels, f"Blinks: {blink_count}", 0, 40)

        if not np.isnan(fixation_index):
            fixation_count = int(fixation_index + 1)
        frame_pixels = write_text(frame_pixels, f"Fixations: {fixation_count}", 0, 80)

        video_time = (frame.time - video_start_time) / 1e9
        video_writer.write_image(frame_pixels, time=video_time)

        cv2.imshow("Frame", frame_pixels)
        cv2.pollKey()

    video_writer.close()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "blinks-and-fixations.mp4")
