import logging
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.timeseries.array_record import Array, Record, fields
from pupil_labs.neon_recording.timeseries.timeseries import Timeseries
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class FixationProps:
    # Note, BlinkProps do not inherit from TimeseriesProps because they should not
    # have a `time` attribute.
    start_time = fields[np.int64]("start_time")  # type:ignore
    "Start timestamp of fixation."

    stop_time = fields[np.int64]("stop_time")  # type:ignore
    "Stop timestamp of fixation."

    start_gaze = fields[np.float32]([
        "start_gaze_x",
        "start_gaze_y",
    ])  # type:ignore
    "Start gaze position in pixels."

    stop_gaze = fields[np.float32]([
        "stop_gaze_x",
        "stop_gaze_y",
    ])  # type:ignore
    "Stop gaze position in pixels."

    mean_gaze = fields[np.float32]([
        "mean_gaze_x",
        "mean_gaze_y",
    ])  # type:ignore
    """Mean gaze position in pixels. Note that this value may be a poor representation
    of the fixation in the presence of VOR movements."""


class FixationRecord(Record, FixationProps):
    def keys(self):
        return [x for x in dir(FixationProps) if not x.startswith("_") and x != "keys"]


class FixationArray(Array[FixationRecord], FixationProps):
    record_class = FixationRecord


class FixationTimeseries(Timeseries[FixationArray, FixationRecord], FixationProps):
    """Fixation event data."""

    name = "fixation"

    def _load_data_from_recording(self, recording: "NeonRecording") -> FixationArray:
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
        data = data[data["event_type"] == 1]
        data = data[
            [
                "start_timestamp_ns",
                "end_timestamp_ns",
                "start_gaze_x",
                "start_gaze_y",
                "end_gaze_x",
                "end_gaze_y",
                "mean_gaze_x",
                "mean_gaze_y",
            ]
        ]
        data.dtype.names = [
            "start_time",
            "stop_time",
            "start_gaze_x",
            "start_gaze_y",
            "stop_gaze_x",
            "stop_gaze_y",
            "mean_gaze_x",
            "mean_gaze_y",
        ]
        data = data.view(FixationArray)
        return data
