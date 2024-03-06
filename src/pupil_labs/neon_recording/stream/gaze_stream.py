import pathlib

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stream import Stream
from ..data_utils import load_with_error_check
from ..time_utils import load_and_convert_tstamps

from .. import structlog
log = structlog.get_logger(__name__)

def _convert_gaze_data_to_recarray(gaze_data, ts, ts_rel):
        log.debug("NeonRecording: Converting gaze data to recarray format.")

        if gaze_data.shape[0] != len(ts):
            log.error("NeonRecording: Length mismatch - gaze_data and ts.")
            raise ValueError("gaze_data and ts must have the same length")
        if len(ts) != len(ts_rel):
            log.error("NeonRecording: Length mismatch - ts and ts_rel.")
            raise ValueError("ts and ts_rel must have the same length")

        out = np.recarray(gaze_data.shape[0], dtype=[('x', '<f8'), ('y', '<f8'), ('ts', '<f8'), ('ts_rel', '<f8')])
        out.x = gaze_data[:, 0]
        out.y = gaze_data[:, 1]
        out.ts = ts.astype(np.float64)
        out.ts_rel = ts_rel.astype(np.float64)
        
        return out


class GazeStream(Stream):
    def to_numpy(self):
        log.debug("NeonRecording: Converting sampled gaze data to numpy array.")

        cid = self.data[self.closest_idxs]
        return pd.DataFrame(cid).to_numpy()


    def load(self, rec_dir: pathlib.Path, start_ts: float) -> None:
        # we use gaze_200hz from cloud for the rec gaze stream
        # ts, raw = self._load_ts_and_data(rec_dir, 'gaze ps1')
        gaze_200hz_ts, gaze_200hz_raw = self._load_ts_and_data(rec_dir, 'gaze_200hz')
        gaze_200hz_ts_rel = gaze_200hz_ts - start_ts

        data = _convert_gaze_data_to_recarray(gaze_200hz_raw, gaze_200hz_ts, gaze_200hz_ts_rel)

        self._data = data
        self.data = self._data[:]
        self.ts = self._data[:].ts
        self.ts_rel = self._data[:].ts_rel

        # because if they use to_numpy without calling sample,
        # then they still get a sensible result: all the data
        self.closest_idxs = np.arange(len(self.data))

        
    def _load_ts_and_data(self, rec_dir: pathlib.Path, stream_name: str):
        log.debug("NeonRecording: Loading gaze data and timestamps.")

        time_path = rec_dir / (stream_name + '.time')
        raw_path = rec_dir / (stream_name + '.raw')

        ts = load_with_error_check(load_and_convert_tstamps, time_path, "Possible error when converting timestamps.")
        raw = load_with_error_check(self._load_raw_data, raw_path, "Please double check the recording download.")

        return ts, raw


    # adapted from @dom:
    # https://github.com/pupil-labs/neon-player/blob/master/pupil_src/shared_modules/pupil_recording/update/neon.py
    def _load_raw_data(self, path: pathlib.Path):
        log.debug("NeonRecording: Loading gaze raw data.")

        raw_data = np.fromfile(str(path), "<f4")
        raw_data.shape = (-1, 2)
        return np.asarray(raw_data, dtype=raw_data.dtype).astype(np.float64)
