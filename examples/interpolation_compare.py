import sys

import cv2
import numpy as np

# Workaround for https://github.com/opencv/opencv/issues/21952
cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

import pupil_labs.neon_recording as nr  # noqa: E402


def make_overlaid_video(recording_dir, output_video_path, fps=None):
    # Open a recording
    recording = nr.open(sys.argv[1])

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (recording.scene.width, recording.scene.height),
    )

    # get closest gaze data to scene frame timestamps
    matched_gazes = recording.gaze.sample(recording.scene.ts)

    # interpolate gaze data to scene frame timestamps
    interpolated_gazes = recording.gaze.interpolate(recording.scene.ts)

    # visualize both
    scene_gaze_pairs = zip(recording.scene, matched_gazes, interpolated_gazes)
    for scene_frame, matched_gaze, interpolated_gaze in scene_gaze_pairs:
        # draw the nearest-time gaze sample in red
        frame = cv2.circle(
            scene_frame.bgr,
            (int(matched_gaze.x), int(matched_gaze.y)),
            50,
            (0, 0, 255),
            10,
        )

        # draw the interpolated gaze sample in blue
        if interpolated_gaze:  # interpolation will fail at the end of the stream
            frame = cv2.circle(
                frame,
                (int(interpolated_gaze.x), int(interpolated_gaze.y)),
                50,
                (255, 0, 0),
                10,
            )

        video_writer.write(frame)
        cv2.imshow("Gaze sample comparison", frame)
        cv2.pollKey()

    cv2.destroyAllWindows()
    video_writer.release()


if __name__ == "__main__":
    make_overlaid_video(sys.argv[1], "interpolation-compare.mp4", 30)
