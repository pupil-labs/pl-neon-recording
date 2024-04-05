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

        gaze_200hz_file = self.recording._rec_dir / "gaze_200hz.raw"
        time_200hz_file = self.recording._rec_dir / "gaze_200hz.time"
        if gaze_200hz_file.exists() and time_200hz_file.exists():
            log.info("NeonRecording: Using 200Hz gaze data")
            gaze_file_pairs.append((gaze_200hz_file, time_200hz_file))

        else:
            log.info("NeonRecording: Using realtime gaze data")
            gaze_file_pairs = find_sorted_multipart_files(self.recording._rec_dir, "gaze")

        gaze_data, time_data = load_multipart_data_time_pairs(gaze_file_pairs, "<f4", 2)
        time_data_rel = time_data - self.recording.start_ts

        self.data = np.rec.fromarrays(
            [time_data, time_data_rel, gaze_data[:,0], gaze_data[:,1]],
            names=["ts", "ts_rel", "x", "y"]
        )

    def _sample_linear_interp(self, sorted_ts):
        result = np.zeros(len(sorted_ts), self.data.dtype).view(np.recarray)

        for key in self.data.dtype.names:
            result[key] = np.interp(sorted_ts, self.ts, self.data[key], left=np.nan, right=np.nan)

        return result
