import pathlib
import numpy as np

# adapted from @dom:
# https://github.com/pupil-labs/neon-player/blob/master/pupil_src/shared_modules/pupil_recording/update/neon.py
def load_raw_data(path: pathlib.Path):
    raw_data = np.fromfile(str(path), "<f4")
    raw_data.shape = (-1, 2)
    return np.asarray(raw_data, dtype=raw_data.dtype).astype(np.float64)


# obtained from @dom:
# https://github.com/pupil-labs/neon-player/blob/master/pupil_src/shared_modules/pupil_recording/update/neon.py
#
# can return none, as worn data is always 1 for Neon
def load_worn_data(path: pathlib.Path):
    if not (path and path.exists()):
        return None

    confidences = np.fromfile(str(path), "<u1") / 255.0
    return np.clip(confidences, 0.0, 1.0)


def convert_gaze_data_to_recarray(gaze_data, ts, ts_rel):
    if gaze_data.shape[0] != len(ts):
        raise ValueError("gaze_data and ts must have the same length")
    if len(ts) != len(ts_rel):
        raise ValueError("ts and ts_rel must have the same length")

    out = np.recarray(gaze_data.shape[0], dtype=[('x', '<f8'), ('y', '<f8'), ('ts', '<f8'), ('ts_rel', '<f8')])
    out.x = gaze_data[:, 0]
    out.y = gaze_data[:, 1]
    out.ts = ts.astype(np.float64)
    out.ts_rel = ts_rel.astype(np.float64)
    
    return out