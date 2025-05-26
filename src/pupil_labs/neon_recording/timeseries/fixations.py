import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.timeseries.array_record import Array, Record, fields
from pupil_labs.neon_recording.timeseries.timeseries import Timeseries, TimeseriesProps
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class FixationProps(TimeseriesProps):
    event_type: npt.NDArray[np.int32] = fields[np.int32]("event_type")  # type:ignore
    "Fixation event kind (0 = saccade / 1 = fixation)"

    start_ts: npt.NDArray[np.int64] = fields[np.int64]("start_timestamp_ns")  # type:ignore
    "Start timestamp of fixation"

    end_ts: npt.NDArray[np.int64] = fields[np.int64]("end_timestamp_ns")  # type:ignore
    "Start timestamp of fixation"

    start_gaze_xy: npt.NDArray[np.float64] = fields[np.float32]([
        "start_gaze_x",
        "start_gaze_y",
    ])  # type:ignore
    "Start gaze position in pixels"

    end_gaze_xy: npt.NDArray[np.float64] = fields[np.float32]([
        "end_gaze_x",
        "end_gaze_y",
    ])  # type:ignore
    "End gaze position in pixels"

    mean_gaze_xy: npt.NDArray[np.float64] = fields[np.float32]([
        "mean_gaze_x",
        "mean_gaze_y",
    ])  # type:ignore
    "Mean gaze position in pixels"

    amplitude_pixels: npt.NDArray[np.float64] = fields[np.float32]("amplitude_pixels")  # type:ignore
    "Amplitude (pixels)"

    amplitude_angle_deg: npt.NDArray[np.float64] = fields[np.float32](
        "amplitude_angle_deg"
    )  # type:ignore
    "Amplitude angle (degrees)"

    mean_velocity: npt.NDArray[np.float64] = fields[np.float32]("mean_velocity")  # type:ignore
    "Mean velocity of fixation (pixels/sec)"

    max_velocity: npt.NDArray[np.float64] = fields[np.float32]("max_velocity")  # type:ignore
    "Max velocity of fixation (pixels/sec)"


class FixationRecord(Record, FixationProps):
    def keys(self):
        return [x for x in dir(FixationProps) if not x.startswith("_") and x != "keys"]


class FixationArray(Array[FixationRecord], FixationProps):
    record_class = FixationRecord


class FixationTimeseries(Timeseries[FixationArray, FixationRecord], FixationProps):
    """Fixation data"""

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
        data = data.view(FixationArray)
        return data
