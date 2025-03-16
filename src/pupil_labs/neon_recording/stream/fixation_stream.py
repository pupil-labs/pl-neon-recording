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


class FixationProps(StreamProps):
    event_type = fields[np.int32]("event_type")
    "Fixation event kind (0 = saccade / 1 = fixation)"

    start_ts = fields[np.int64]("start_timestamp_ns")
    "Start timestamp of fixation"

    end_ts = fields[np.int64]("end_timestamp_ns")
    "Start timestamp of fixation"

    start_gaze_xy = fields[np.float32](["start_gaze_x", "start_gaze_y"])
    "Start gaze position in pixels"

    end_gaze_xy = fields[np.float32](["end_gaze_x", "end_gaze_y"])
    "End gaze position in pixels"

    mean_gaze_xy = fields[np.float32](["mean_gaze_x", "mean_gaze_y"])
    "Mean gaze position in pixels"

    amplitude_pixels = fields[np.float32]("amplitude_pixels")
    "Amplitude (pixels)"

    amplitude_angle_deg = fields[np.float32]("amplitude_angle_deg")
    "Amplitude angle (degrees)"

    mean_velocity = fields[np.float32]("mean_velocity")
    "Mean velocity of fixation (pixels/sec)"

    max_velocity = fields[np.float32]("max_velocity")
    "Max velocity of fixation (pixels/sec)"


class FixationRecord(Record, FixationProps):
    def keys(self):
        keys = FixationProps.__dict__.keys()
        return [x for x in keys if not x.startswith("_")]


class FixationArray(Array[FixationRecord], FixationProps):
    record_class = FixationRecord


class FixationStream(Stream[FixationArray, FixationRecord], FixationProps):
    """Fixation data"""

    data: FixationArray

    def __init__(self, recording: "NeonRecording"):
        log.debug("NeonRecording: Loading fixation data")
        file_pairs = find_sorted_multipart_files(recording._rec_dir, "fixations")
        data = load_multipart_data_time_pairs(
            file_pairs,
            np.dtype([
                ("event_type", "int32"),
                ("start_timestamp_ns", "int64"),
                ("end_timestamp_ns", "int64"),
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
            ]),
        )
        super().__init__("fixation", recording, data.view(FixationArray))
