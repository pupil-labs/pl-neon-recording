from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.constants import TIMESTAMP_DTYPE
from pupil_labs.neon_recording.stream.array_record import (
    Array,
    Record,
    join_struct_arrays,
    proxy,
)
from pupil_labs.neon_recording.utils import find_sorted_multipart_files

from .. import structlog
from .stream import Stream, StreamProps

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class GazeProps(StreamProps):
    x = proxy[np.float64]("x")
    "Gaze x coordinate in pixels"

    y = proxy[np.float64]("y")
    "Gaze y coordinate in pixels"

    xy = proxy[np.float64](["x", "y"])
    "Gaze xy coordinates in pixels"


class GazeRecord(Record, GazeProps):
    def keys(self):
        return [x for x in GazeProps.__dict__.keys() if not x.startswith("_")]


class GazeArray(Array[GazeRecord], GazeProps):
    record_class = GazeRecord


class GazeStream(Stream[GazeRecord], GazeProps):
    """
    Gaze data
    """

    data: GazeArray

    def __init__(self, recording: "NeonRecording"):
        log.debug("NeonRecording: Loading eye state data")

        gaze_200hz_file = recording._rec_dir / "gaze_200hz.raw"
        time_200hz_file = recording._rec_dir / "gaze_200hz.time"

        gaze_file_pairs = []
        if gaze_200hz_file.exists() and time_200hz_file.exists():
            log.debug("NeonRecording: Using 200Hz gaze data")
            gaze_file_pairs.append((gaze_200hz_file, time_200hz_file))
        else:
            log.debug("NeonRecording: Using realtime gaze data")
            gaze_file_pairs = find_sorted_multipart_files(recording._rec_dir, "gaze")

        time_data = Array([file for _, file in gaze_file_pairs], TIMESTAMP_DTYPE)
        gaze_data = Array(
            [file for file, _ in gaze_file_pairs],
            fallback_dtype=np.dtype(
                [
                    ("x", "float32"),
                    ("y", "float32"),
                ]
            ),
        )
        data = join_struct_arrays([time_data, gaze_data]).view(GazeArray)
        super().__init__("gaze", recording, data)
