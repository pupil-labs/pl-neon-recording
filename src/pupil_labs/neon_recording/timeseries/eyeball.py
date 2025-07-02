import logging
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.timeseries.array_record import Array, Record, fields
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
    from ..neon_recording import NeonRecording


class EyeballProps(TimeseriesProps):
    center_left = fields[np.float64]([
        "eyeball_center_left_x",
        "eyeball_center_left_y",
        "eyeball_center_left_z",
    ])  # type:ignore
    "The 3D position of the left eyeball relative to the scene camera in mm."

    center_right = fields[np.float64]([
        "eyeball_center_right_x",
        "eyeball_center_right_y",
        "eyeball_center_right_z",
    ])  # type:ignore
    "The 3D position of the right eyeball relative to the scene camera in mm."

    optical_axis_left = fields[np.float64]([
        "optical_axis_left_x",
        "optical_axis_left_y",
        "optical_axis_left_z",
    ])  # type:ignore
    "A 3D vector in the forward direction of the left eye's optical axis"

    optical_axis_right = fields[np.float64]([
        "optical_axis_right_x",
        "optical_axis_right_y",
        "optical_axis_right_z",
    ])  # type:ignore
    "A 3D vector in the forward direction of the right eye's optical axis"


class EyeballRecord(Record, EyeballProps):
    def keys(self):
        return [x for x in dir(EyeballProps) if not x.startswith("_") and x != "keys"]


class EyeballArray(Array[EyeballRecord], EyeballProps):
    record_class = EyeballRecord


class EyeballTimeseries(
    InterpolatableTimeseries[EyeballArray, EyeballRecord], EyeballProps
):
    """Eyeball data"""

    name: str = "eyeball"

    def _load_data_from_recording(self, recording: "NeonRecording") -> EyeballArray:
        log.debug("NeonRecording: Loading eye state data")
        file_pairs = find_sorted_multipart_files(recording._rec_dir, "eye_state")
        data = load_multipart_data_time_pairs(
            file_pairs,
            dtype=np.dtype([
                ("pupil_diameter_left_mm", "float32"),
                ("eyeball_center_left_x", "float32"),
                ("eyeball_center_left_y", "float32"),
                ("eyeball_center_left_z", "float32"),
                ("optical_axis_left_x", "float32"),
                ("optical_axis_left_y", "float32"),
                ("optical_axis_left_z", "float32"),
                ("pupil_diameter_right_mm", "float32"),
                ("eyeball_center_right_x", "float32"),
                ("eyeball_center_right_y", "float32"),
                ("eyeball_center_right_z", "float32"),
                ("optical_axis_right_x", "float32"),
                ("optical_axis_right_y", "float32"),
                ("optical_axis_right_z", "float32"),
            ]),
        )
        data = data[
            [
                "time",
                "eyeball_center_left_x",
                "eyeball_center_left_y",
                "eyeball_center_left_z",
                "optical_axis_left_x",
                "optical_axis_left_y",
                "optical_axis_left_z",
                "eyeball_center_right_x",
                "eyeball_center_right_y",
                "eyeball_center_right_z",
                "optical_axis_right_x",
                "optical_axis_right_y",
                "optical_axis_right_z",
            ]
        ]
        data.dtype.names = (
            "time",
            "center_left_x",
            "center_left_y",
            "center_left_z",
            "optical_axis_left_x",
            "optical_axis_left_y",
            "optical_axis_left_z",
            "center_right_x",
            "center_right_y",
            "center_right_z",
            "optical_axis_right_x",
            "optical_axis_right_y",
            "optical_axis_right_z",
        )
        data = data.view(EyeballArray)
        return data
