from logging import getLogger
from pathlib import Path

import numpy as np

from pupil_labs.matching import Sensor

from ..utils import find_sorted_multipart_files, load_multipart_data_time_pairs

log = getLogger(__name__)


class EyeState(np.ndarray, Sensor):
    def __new__(cls, rec_dir: Path):
        log.debug("NeonRecording: Loading eye state data")

        eye_state_files = find_sorted_multipart_files(rec_dir, "eye_state")
        eye_state_data, time_data = load_multipart_data_time_pairs(
            eye_state_files, "<f4", 2
        )

        data = eye_state_data.reshape(-1, 14)
        data = np.vstack([time_data, data.T])
        data = np.rec.fromarrays(
            data,
            names=[
                "ts",
                "pupil_diameter_left",
                "eyeball_center_left_x",
                "eyeball_center_left_y",
                "eyeball_center_left_z",
                "optical_axis_left_x",
                "optical_axis_left_y",
                "optical_axis_left_z",
                "pupil_diameter_right",
                "eyeball_center_right_x",
                "eyeball_center_right_y",
                "eyeball_center_right_z",
                "optical_axis_right_x",
                "optical_axis_right_y",
                "optical_axis_right_z",
            ],
        )
        data = data.view(cls)

        data.timestamps = data["ts"]
        return data
