import sys

import numpy as np

import pupil_labs.neon_recording as nr


def find_clap(recording_dir):
    recording = nr.load(recording_dir)

    max_rms = -1
    max_time = -1

    for frame in recording.audio:
        audio_data = frame.to_ndarray()
        rms = np.sqrt(np.mean(audio_data**2))

        if rms > max_rms:
            max_rms = rms
            max_time = frame.ts

    rel_time = (max_time - recording.start_ts) / 1e9
    print(
        f"The loudest audio occurs at {max_time} (rel={rel_time}) with an rms = {max_rms}."
    )


if __name__ == "__main__":
    find_clap(sys.argv[1])
