import pathlib

import numpy as np

from .. import structlog
from ..utils import find_sorted_multipart_files, load_multipart_data_time_pairs
from .stream import Stream

log = structlog.get_logger(__name__)


class GazeStream(Stream):
    def __init__(self, name, recording):
        super().__init__(name, recording)

        log.info("NeonRecording: Loading gaze data")

        gaze_file_pairs = []

        gaze_200hz_file = self._recording._rec_dir / "gaze_200hz.raw"
        time_200hz_file = self._recording._rec_dir / "gaze_200hz.time"
        if gaze_200hz_file.exists() and time_200hz_file.exists():
            log.info("NeonRecording: Using 200Hz gaze data")
            gaze_file_pairs.append((gaze_200hz_file, time_200hz_file))

        else:
            log.info("NeonRecording: Using realtime gaze data")
            gaze_file_pairs = find_sorted_multipart_files(self._recording._rec_dir, "gaze")

        gaze_data, time_data = load_multipart_data_time_pairs(gaze_file_pairs, "<f4", 2)
        rel_time_data = time_data - self._recording.start_ts

        self._data = np.recarray(
            gaze_data.shape[0],
            dtype=[("x", "<f8"), ("y", "<f8"), ("ts", "<f8"), ("ts_rel", "<f8")],
        )
        self._data.x = gaze_data[:, 0]
        self._data.y = gaze_data[:, 1]
        self._data.ts = time_data.astype(np.float64)
        self._data.ts_rel = rel_time_data.astype(np.float64)

        self._ts = self._data[:].ts


    def _sample_linear_interp(self, sorted_ts):
        xs = self._data.x
        ys = self._data.y

        interp_data = np.zeros(
            len(sorted_ts),
            dtype=[("x", "<f8"), ("y", "<f8"), ("ts", "<f8"), ("ts_rel", "<f8")],
        ).view(np.recarray)
        interp_data.x = np.interp(sorted_ts, self._ts, xs, left=np.nan, right=np.nan)
        interp_data.y = np.interp(sorted_ts, self._ts, ys, left=np.nan, right=np.nan)
        interp_data.ts = sorted_ts
        interp_data.ts_rel = np.interp(
            sorted_ts, self._ts, self._ts_rel, left=np.nan, right=np.nan
        )

        for d in interp_data:
            if not np.isnan(d.x):
                yield d
            else:
                yield None
