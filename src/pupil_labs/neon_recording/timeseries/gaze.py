import logging
from typing import (
    TYPE_CHECKING,
)

import numpy as np

from pupil_labs.neon_recording.timeseries.array_record import (
    Array,
    Record,
    fields,
)
from pupil_labs.neon_recording.timeseries.timeseries import (
    InterpolatableTimeseries,
    TimeseriesProps,
)
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class GazeProps(TimeseriesProps):
    point = fields[np.float64](["point_x", "point_x"])  # type:ignore
    "2D gaze coordinates in the scene video in pixels"


class GazeRecord(Record, GazeProps):
    def keys(self):
        return [x for x in dir(GazeProps) if not x.startswith("_") and x != "keys"]


class GazeArray(Array[GazeRecord], GazeProps):
    record_class = GazeRecord


class GazeTimeseries(InterpolatableTimeseries[GazeArray, GazeRecord], GazeProps):
    name: str = "gaze"

    def _load_data_from_recording(self, recording) -> "GazeArray":
        log.debug("NeonRecording: Loading gaze data")

        gaze_200hz_file = recording._rec_dir / "gaze_200hz.raw"
        time_200hz_file = recording._rec_dir / "gaze_200hz.time"

        file_pairs = []
        if gaze_200hz_file.exists() and time_200hz_file.exists():
            log.debug("NeonRecording: Using 200Hz gaze data")
            file_pairs.append((gaze_200hz_file, time_200hz_file))
        else:
            log.debug("NeonRecording: Using realtime gaze data")
            file_pairs = find_sorted_multipart_files(recording._rec_dir, "gaze")

        data = load_multipart_data_time_pairs(
            file_pairs,
            np.dtype([
                ("x", "float32"),
                ("y", "float32"),
            ]),
        )
        data.dtype.names = ("time", "point_x", "point_y")
        data = data.view(GazeArray)
        return data
