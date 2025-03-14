import logging
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.stream.array_record import Array, Record, fields
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

from .stream import Stream, StreamProps

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class GazeProps(StreamProps):
    x = fields[np.float64]("x")
    "Gaze x coordinate in pixels"

    y = fields[np.float64]("y")
    "Gaze y coordinate in pixels"

    xy = fields[np.float64](["x", "y"])
    "Gaze xy coordinates in pixels"


class GazeRecord(Record, GazeProps):
    def keys(self):
        keys = GazeProps.__dict__.keys()
        return [x for x in keys if not x.startswith("_")]


class GazeArray(Array[GazeRecord], GazeProps):
    record_class = GazeRecord


class GazeStream(Stream[GazeRecord], GazeProps):
    """Gaze data"""

    data: GazeArray

    def __init__(self, recording: "NeonRecording"):
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
        super().__init__("gaze", recording, data.view(GazeArray))
