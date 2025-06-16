import sys

import numpy as np
from tqdm import tqdm

import pupil_labs.neon_recording as nr


def find_clap(recording_dir, window_size_seconds=0.1, first_n_seconds=10):
    recording = nr.open(recording_dir)

    audio_data = np.array([], dtype=np.float32)
    ts_lookup = []
    for frame in recording.audio:
        if first_n_seconds is not None:
            rel_time = (frame.ts - recording.start_ts) / 1e9
            if rel_time > first_n_seconds:
                break

        # Create a timestamp lookup table
        ts_lookup.append([len(audio_data), frame.ts])

        # Gather all the audio samples
        audio_data = np.concatenate((audio_data, frame.to_ndarray().flatten()))

    ts_lookup = np.array(ts_lookup)

    # Calculate RMS over a sliding window
    # Remember the sample index of the loudest window
    max_rms = 0
    loudest_sample_idx = 0

    samples_per_window = int(window_size_seconds * recording.audio.rate)
    for i in tqdm(range(len(audio_data) - samples_per_window)):
        segment = audio_data[i : i + samples_per_window]
        rms = np.sqrt(np.mean(np.square(segment)))

        if rms > max_rms:
            max_rms = rms
            loudest_sample_idx = int(i + samples_per_window / 2)

    # Find the reference timestamp from the lookup table
    lookup_idx = np.searchsorted(ts_lookup[:, 0], loudest_sample_idx) - 1
    reference_ts = ts_lookup[lookup_idx, 1]

    # Calculate the sample timestamp using the reference timestamp
    samples_after_reference = loudest_sample_idx - ts_lookup[lookup_idx, 0]
    loudest_time = reference_ts + (samples_after_reference / recording.audio.rate) * 1e9

    print(f"The loudest audio occurs at {loudest_time:.0f} rms = {max_rms:.3f}.")
    print(
        f"    Relative to recording start: "
        f"{(loudest_time - recording.start_ts) / 1e9:0.3f}s"
    )
    print(
        f"    Relative to video start    : "
        f"{(loudest_time - recording.scene.ts[0]) / 1e9:0.3f}s"
    )
    print(
        f"    Relative to audio start    : "
        f"{(loudest_time - recording.audio.ts[0]) / 1e9:0.3f}s"
    )


if __name__ == "__main__":
    find_clap(sys.argv[1])
