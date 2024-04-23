import sys
from fractions import Fraction

import cv2
import numpy as np
import pupil_labs.neon_recording as nr
import pupil_labs.video as plv

from pupil_labs.neon_recording.stream.av_stream.video_stream import GrayFrame

from tqdm import tqdm

def make_overlaid_video(recording_dir, output_video_path, fps=30):
    recording = nr.load(recording_dir)

    output_container = plv.open(output_video_path, mode="w")
    out_video_stream = output_container.add_stream("libx264", rate=fps)
    out_video_stream.width = 1600
    out_video_stream.height = 1200
    out_video_stream.time_base = Fraction(1, fps)

    output_timestamps = np.arange(recording.scene.ts[0], recording.scene.ts[-1], 1/fps)

    combined_data = zip(
        output_timestamps,
        recording.scene.sample(output_timestamps),
        recording.gaze.sample(output_timestamps),
    )

    frame_idx = 0
    for ts, scene_frame, gaze_datum in tqdm(combined_data, total=len(output_timestamps)):
        frame_idx += 1
        if abs(scene_frame.ts - ts) < 2/fps:
            frame_pixels = scene_frame.bgr
        else:
            frame_pixels = GrayFrame(out_video_stream.width, out_video_stream.height).bgr

        if gaze_datum is not None and not np.isnan(gaze_datum.x):
            if abs(gaze_datum.ts - ts) < 2/fps:
                frame_pixels = cv2.circle(
                    frame_pixels,
                    (int(gaze_datum.x), int(gaze_datum.y)),
                    50,
                    (0, 0, 255),
                    10
                )

        output_frame = plv.VideoFrame.from_ndarray(frame_pixels, format='bgr24')
        output_frame.pts = frame_idx

        for packet in out_video_stream.encode(output_frame):
            output_container.mux(packet)

    for packet in out_video_stream.encode():
        output_container.mux(packet)

    output_container.close()


if __name__ == '__main__':
    make_overlaid_video(sys.argv[1], "gaze-overlay-output-video.mp4", 24)
