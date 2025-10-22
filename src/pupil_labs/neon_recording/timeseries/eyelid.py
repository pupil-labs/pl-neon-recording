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


class EyelidProps(TimeseriesProps):
    angle_left = fields[np.float64]([
        "angle_upper_left",
        "angle_lower_left",
    ])
    "Opening angles of the upper and lower eyelids of the left eye."

    angle_right = fields[np.float64]([
        "angle_upper_right",
        "angle_lower_right",
    ])
    "Opening angles of the upper and lower eyelids of the right eye."

    aperture_left = fields[np.float64](["aperture_left"])
    "Eyelid aperture of the left eye in mm."

    aperture_right = fields[np.float64](["aperture_right"])
    "Eyelid aperture of the right eye in mm."


class EyelidRecord(Record, EyelidProps):
    def keys(self):
        return [x for x in dir(EyelidProps) if not x.startswith("_") and x != "keys"]


class EyelidArray(Array[EyelidRecord], EyelidProps):
    record_class = EyelidRecord


class EyelidTimeseries(
    InterpolatableTimeseries[EyelidArray, EyelidRecord], EyelidProps
):
    """Eyelid data describing the opening angles and apertures of each eyelid."""

    name: str = "eyelid"

    def _load_data_from_recording(self, recording: "NeonRecording") -> EyelidArray:
        log.debug("NeonRecording: Loading eye state data")
        file_pairs = find_sorted_multipart_files(recording._rec_dir, "eye_state")

        if len(file_pairs) == 0:
            raise AttributeError("No eyelid data found")

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
                "eyelid_angle_top_left",
                "eyelid_angle_bottom_left",
                "eyelid_aperture_left_mm",
                "eyelid_angle_top_right",
                "eyelid_angle_bottom_right",
                "eyelid_aperture_right_mm",
            ]
        ]
        data.dtype.names = (
            "time",
            "angle_upper_left",
            "angle_lower_left",
            "aperture_left",
            "angle_upper_right",
            "angle_lower_right",
            "aperture_right",
        )
        data = data.view(EyelidArray)
        return data  # type: ignore
