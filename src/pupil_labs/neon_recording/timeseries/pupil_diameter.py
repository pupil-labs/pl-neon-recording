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


class PupilDiameterProps(TimeseriesProps):
    left = fields[np.float64](["left"])
    "Pupil diameter of the left eye in mm."

    right = fields[np.float64](["right"])
    "Pupil diameter of the right eye in mm."


class PupilDiameterRecord(Record, PupilDiameterProps):
    def keys(self):
        return [
            x for x in dir(PupilDiameterProps) if not x.startswith("_") and x != "keys"
        ]


class PupilDiameterArray(Array[PupilDiameterRecord], PupilDiameterProps):
    record_class = PupilDiameterRecord


class PupilDiameterTimeseries(
    Timeseries[PupilDiameterArray, PupilDiameterRecord], PupilDiameterProps
):
    """Pupil diameter data"""

    name: str = "pupil_diameter"

    def _load_data_from_recording(
        self, recording: "NeonRecording"
    ) -> PupilDiameterArray:
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
                "pupil_diameter_left_mm",
                "pupil_diameter_right_mm",
            ]
        ]
        data.dtype.names = (
            "left",
            "right",
        )
        data = data.view(PupilDiameterArray)

        return data
