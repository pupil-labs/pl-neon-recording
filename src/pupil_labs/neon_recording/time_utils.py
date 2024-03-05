import pathlib

import numpy as np

# adapted from @dom:
# https://github.com/pupil-labs/neon-player/blob/master/pupil_src/shared_modules/pupil_recording/update/neon.py
def ns_to_s(timestamps):
    SECONDS_PER_NANOSECOND = 1e-9
    return timestamps * SECONDS_PER_NANOSECOND


# adapted from @dom and @pfaion:
# https://github.com/pupil-labs/neon-player/blob/master/pupil_src/shared_modules/pupil_recording/update/update_utils.py
def rewrite_times(path: pathlib.Path, dtype="<u8"):
    """Load raw times (assuming dtype), apply conversion and save as _timestamps.npy."""
    
    timestamps = np.fromfile(str(path), dtype=dtype)
    new_name = f"{path.stem}_timestamps_unix.npy"
    timestamp_loc = path.parent / new_name
    np.save(str(timestamp_loc), timestamps)

    timestamps = ns_to_s(timestamps)

    new_name = f"{path.stem}_timestamps.npy"
    timestamp_loc = path.parent / new_name
    np.save(str(timestamp_loc), timestamps)


# obtained from @dom:
# https://github.com/pupil-labs/neon-player/blob/master/pupil_src/shared_modules/pupil_recording/update/neon.py
def load_timestamps_data(path: pathlib.Path):
    timestamps = np.load(str(path))
    return timestamps.astype(np.float64)


# obtained from @dom:
# https://github.com/pupil-labs/neon-player/blob/master/pupil_src/shared_modules/pupil_recording/update/neon.py
def neon_raw_time_load(path: pathlib.Path):
    return np.fromfile(str(path), dtype="<u8")