import sys
from tqdm import tqdm

import cv2
import numpy as np

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import pupil_labs.neon_recording as nr # noqa


class EventTracker:
    def __init__(self, stream):
        self.idx = 0
        self.itr = iter(stream)
        self.next_event = next(self.itr)

    def in_event(self, ts):
        if self.next_event is None:
            return False

        return self.next_event.start_timestamp_ns < ts < self.next_event.end_timestamp_ns

    def step_to(self, ts):
        try:
            while self.next_event is not None and self.next_event.end_timestamp_ns < ts:
                self.next_event = next(self.itr)
                self.idx += 1

        except StopIteration:
            self.next_event = None

        return self.in_event(ts)


def write_text(image, text, x, y):
    return cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 2, cv2.LINE_AA,
    )


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

    fixations_only = recording.fixations[recording.fixations["event_type"] == 1]
    event_trackers = {
        "Fixation": EventTracker(fixations_only),
        "Blink": EventTracker(recording.blinks),
    }

    for frame in tqdm(eye_frames):
        frame_pixels = frame.bgr

        text_y = 0
        for stream, tracker in event_trackers.items():
            text_y += 40
            if tracker.step_to(frame.ts):
                frame_pixels = write_text(
                    frame_pixels,
                    f"{stream} {tracker.idx + 1}",
                    0, text_y
                )

        video_writer.write(frame_pixels)
        cv2.imshow("Frame", frame_pixels)
        cv2.pollKey()

    video_writer.release()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "blinks-and-fixations.mp4")
