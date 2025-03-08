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


class FixationProps(StreamProps):
    event_type = proxy[np.int32]("event_type")
    "Fixation event kind (saccade / fixation)"

    start_ts = proxy[np.int64]("start_time_ns")
    "Start timestamp of fixation"

    end_ts = proxy[np.int64]("end_time_ns")
    "Start timestamp of fixation"

    start_gaze_xy = proxy[np.float32](["start_gaze_x", "start_gaze_y"])
    "Start gaze position in pixels"

    end_gaze_xy = proxy[np.float32](["end_gaze_x", "end_gaze_y"])
    "End gaze position in pixels"

    mean_gaze_xy = proxy[np.float32](["mean_gaze_x", "mean_gaze_y"])
    "Mean gaze position in pixels"

    amplitude_pixels = proxy[np.float32]("amplitude_pixels")
    "Amplitude (pixels)"

    amplitude_angle_deg = proxy[np.float32]("amplitude_angle_deg")
    "Amplitude angle (degrees)"

    mean_velocity = proxy[np.float32]("mean_velocity")
    "Mean velocity of fixation (pixels/sec)"

    max_velocity = proxy[np.float32]("max_velocity")
    "Max velocity of fixation (pixels/sec)"


class FixationRecord(Record, FixationProps):
    def keys(self):
        return [x for x in FixationProps.__dict__.keys() if not x.startswith("_")]


class FixationArray(Array[FixationRecord], FixationProps):
    record_class = FixationRecord


class FixationStream(Stream[FixationRecord], FixationProps):
    """
    Fixation data
    """

    data: FixationArray

    def __init__(self, recording: "NeonRecording"):
        log.debug("NeonRecording: Loading fixation data")
        fixation_file_pairs = find_sorted_multipart_files(
            recording._rec_dir, "fixations"
        )

        time_data = Array([file for _, file in fixation_file_pairs], TIMESTAMP_DTYPE)
        fixation_data = Array(
            [file for file, _ in fixation_file_pairs],
            fallback_dtype=np.dtype(
                [
                    ("event_type", "int32"),
                    ("start_time_ns", "int64"),
                    ("end_time_ns", "int64"),
                    ("start_gaze_x", "float32"),
                    ("start_gaze_y", "float32"),
                    ("end_gaze_x", "float32"),
                    ("end_gaze_y", "float32"),
                    ("mean_gaze_x", "float32"),
                    ("mean_gaze_y", "float32"),
                    ("amplitude_pixels", "float32"),
                    ("amplitude_angle_deg", "float32"),
                    ("mean_velocity", "float32"),
                    ("max_velocity", "float32"),
                ]
            ),
        )
        data = join_struct_arrays([time_data, fixation_data]).view(FixationArray)
        super().__init__("fixation", recording, data)
