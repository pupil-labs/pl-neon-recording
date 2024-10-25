from logging import getLogger
from pathlib import Path

import numpy as np

from pupil_labs.matching import Sensor

from ..utils import find_sorted_multipart_files, load_multipart_data_time_pairs

log = getLogger(__name__)


class Gaze(np.ndarray, Sensor):
    def __new__(cls, rec_dir: Path):
        log.debug("NeonRecording: Loading gaze data")

        gaze_file_pairs = []

        gaze_200hz_file = rec_dir / "gaze_200hz.raw"
        time_200hz_file = rec_dir / "gaze_200hz.time"
        if gaze_200hz_file.exists() and time_200hz_file.exists():
            log.debug("NeonRecording: Using 200Hz gaze data")
            gaze_file_pairs.append((gaze_200hz_file, time_200hz_file))

        else:
            log.debug("NeonRecording: Using realtime gaze data")
            gaze_file_pairs = find_sorted_multipart_files(rec_dir, "gaze")

        gaze_data, time_data = load_multipart_data_time_pairs(gaze_file_pairs, "<f4", 2)

        data = np.rec.fromarrays(
            [time_data, gaze_data[:, 0], gaze_data[:, 1]], names=["ts", "x", "y"]
        )
        data = data.view(cls)

        data.timestamps = data["ts"]
        return data
