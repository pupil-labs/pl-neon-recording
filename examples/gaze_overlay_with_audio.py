import sys

import numpy as np
import cv2
import av

import pupil_labs.neon_recording as nr
from pupil_labs.neon_recording.stream.av_stream.video_stream import GrayFrame

from tqdm import tqdm


def make_overlaid_video(recording_dir, output_video_path, fps=None):
    recording = nr.load(recording_dir)

    output_container = av.open(str(output_video_path), 'w')

    # create video output stream
    input_video_stream = recording.scene.av_streams[0]
    output_video_stream = output_container.add_stream(
        input_video_stream.name,
        rate=input_video_stream.average_rate,
        options=input_video_stream.options,
        time_base=input_video_stream.time_base,
    )

    # create audio output stream
    input_audio_stream = recording.audio.av_streams[0]
    output_audio_stream = output_container.add_stream(
        input_audio_stream.name,
        rate=input_audio_stream.rate,
        format=input_audio_stream.format,
        layout=input_audio_stream.layout,
        time_base=input_audio_stream.time_base,
        options=input_audio_stream.options,
    )

    if fps is None:
        output_timestamps = recording.scene.ts
    else:
        output_timestamps = np.arange(recording.scene.ts[0], recording.scene.ts[-1], 1 / fps)

    combined_data = zip(
        output_timestamps,
        recording.scene.sample(output_timestamps),
        recording.gaze.sample(output_timestamps),
    )

    # To gaurantee that no audio frames are skipped, do not sample the audio stream with video stream timestamps
    # Instead, find all the audio stream timestamps that exist between the desired start- and end-timestamps
    # Then use these to sample the audio stream
    audio_ts_filter = (recording.audio.ts >= output_timestamps[0]) & (recording.audio.ts <= output_timestamps[-1])
    audio_timestamps = recording.audio.ts[audio_ts_filter]
    audio_samples = recording.audio.sample(audio_timestamps)

    # Audio and video frames aren't necessarily 1-to-1, so we need to iterate them separately
    audio_iterator = iter(audio_samples)

    first_video_frame = None
    first_audio_frame = None
    audio_frame = None

    for ts, scene_frame, gaze_datum in tqdm(combined_data, total=len(output_timestamps)):
        if first_video_frame is None:
            first_video_frame = scene_frame

        if abs(scene_frame.ts - ts) < 2 / input_video_stream.base_rate:
            frame_pixels = scene_frame.bgr
        else:
            frame_pixels = GrayFrame(scene_frame.width, scene_frame.height).bgr

        # draw gaze circle
        if abs(gaze_datum.ts - ts) < 2 / input_video_stream.base_rate:
            frame_pixels = cv2.circle(
                frame_pixels,
                (int(gaze_datum.x), int(gaze_datum.y)),
                50,
                (0, 0, 255),
                10
            )

        # encode video
        output_video_frame = av.VideoFrame.from_ndarray(frame_pixels, format='bgr24')
        output_video_frame.time_base = scene_frame.time_base

        output_video_frame.pts = (scene_frame.ts - first_video_frame.ts) / output_video_frame.time_base
        for packet in output_video_stream.encode(output_video_frame):
            output_container.mux(packet)

        # encode audio
        for audio_frame in audio_iterator:
            if first_audio_frame is None:
                first_audio_frame = audio_frame

            audio_data = audio_frame.to_ndarray()
            output_audio_frame = av.AudioFrame.from_ndarray(audio_data, format=audio_frame.format.name, layout=audio_frame.layout)
            output_audio_frame.time_base = audio_frame.time_base
            output_audio_frame.rate = audio_frame.rate
            output_audio_frame.pts = round((audio_frame.ts - first_audio_frame.ts) / output_audio_frame.time_base)

            for packet in output_audio_stream.encode(output_audio_frame):
                output_container.mux(packet)

            duration = output_audio_frame.samples / output_audio_frame.rate
            # Check if the audio stream has caught up to the video stream
            # if so, move on to the next video frame
            if audio_frame.ts + duration >= scene_frame.ts:
                break

    # flush and close streams
    for stream in [output_video_stream, output_audio_stream]:
        for packet in stream.encode(None):
            output_container.mux(packet)

        stream.close()

    # close container
    output_container.close()


if __name__ == '__main__':
    make_overlaid_video(sys.argv[1], "gaze-overlay-output-video.mkv")
