import sys

import numpy as np
from tqdm import tqdm

import pupil_labs.neon_recording as nr


def find_clap(recording_dir, window_size_seconds=0.1):
    recording = nr.open(recording_dir)
    sample_count = int(window_size_seconds * recording.audio.rate)

    audio_frames = [frame.to_ndarray() for frame in recording.audio]
    audio_data = np.concatenate(audio_frames).flatten()

    max_rms = 0
    loudest_time = 0

    for i in tqdm(range(len(audio_data) - sample_count)):
        segment = audio_data[i:i + sample_count]
        rms = np.sqrt(np.mean(np.square(segment)))

        if rms > max_rms:
            max_rms = rms
            loudest_time = float(i) / recording.audio.rate

    abs_time = loudest_time * 1e9 + recording.start_ts
    print(f"The loudest audio occurs at {abs_time} (rel={loudest_time}), rms = {max_rms}.")


if __name__ == "__main__":
    find_clap(sys.argv[1])
