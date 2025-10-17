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


class PupilProps(TimeseriesProps):
    diameter_left = fields[np.float64](["diameter_left"])
    "Pupil diameter of the left eye in mm."

    diameter_right = fields[np.float64](["diameter_right"])
    "Pupil diameter of the right eye in mm."


class PupilRecord(Record, PupilProps):
    def keys(self):
        return [x for x in dir(PupilProps) if not x.startswith("_") and x != "keys"]


class PupilArray(Array[PupilRecord], PupilProps):
    record_class = PupilRecord


class PupilTimeseries(InterpolatableTimeseries[PupilArray, PupilRecord], PupilProps):
    """Pupil data"""

    name: str = "pupil"

    def _load_data_from_recording(self, recording: "NeonRecording") -> PupilArray:
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
                "pupil_diameter_left_mm",
                "pupil_diameter_right_mm",
            ]
        ]
        data.dtype.names = (
            "time",
            "diameter_left",
            "diameter_right",
        )
        data = data.view(PupilArray)

        return data  # type: ignore
