import logging
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.timeseries.array_record import Array, Record, fields
from pupil_labs.neon_recording.timeseries.timeseries import Timeseries, TimeseriesProps
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class SaccadeProps(TimeseriesProps):
    start_time = fields[np.int64]("start_time")  # type:ignore
    "Start timestamp of Saccade"

    end_time = fields[np.int64]("end_time")  # type:ignore
    "End timestamp of Saccade"

    start_gaze = fields[np.float32]([
        "start_gaze_x",
        "start_gaze_y",
    ])  # type:ignore
    "Start gaze position in pixels"

    stop_gaze = fields[np.float32]([
        "stop_gaze_x",
        "stop_gaze_y",
    ])  # type:ignore
    "Stop gaze position in pixels"

    mean_gaze = fields[np.float32]([
        "mean_gaze_x",
        "mean_gaze_y",
    ])  # type:ignore
    "Mean gaze position in pixels"

    amplitude = fields[np.float32]("amplitude_angle")  # type:ignore
    "Amplitude angle (degrees)"

    mean_velocity = fields[np.float32]("mean_velocity")  # type:ignore
    "Mean velocity of Saccade (pixels/sec)"

    max_velocity = fields[np.float32]("max_velocity")  # type:ignore
    "Max velocity of Saccade (pixels/sec)"


class SaccadeRecord(Record, SaccadeProps):
    def keys(self):
        return [x for x in dir(SaccadeProps) if not x.startswith("_") and x != "keys"]


class SaccadeArray(Array[SaccadeRecord], SaccadeProps):
    record_class = SaccadeRecord


class SaccadeTimeseries(Timeseries[SaccadeArray, SaccadeRecord], SaccadeProps):
    """Saccade data"""

    name = "saccade"

    def _load_data_from_recording(self, recording: "NeonRecording") -> SaccadeArray:
        log.debug("NeonRecording: Loading Saccade data")
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
        data = data[data["event_type"] == 0]
        data = data[["start_timestamp_ns", "end_timestamp_ns", "start_gaze_x", "start_gaze_y", "end_gaze_x", "end_gaze_y", "amplitude_angle_deg", "mean_velocity", "max_velocity"]]
        data.dtype.names = (
            "start_time",
            "end_time",
            "start_gaze_x",
            "start_gaze_y",
            "stop_gaze_x",
            "stop_gaze_y",
            "amplitude_angle",
            "mean_velocity",
            "max_velocity",
        )
        data = data.view(SaccadeArray)
        return data
