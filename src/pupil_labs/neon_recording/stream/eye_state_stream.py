import numpy as np

from .. import structlog
from .stream import Stream
from ..utils import find_sorted_multipart_files, load_multipart_data_time_pairs

log = structlog.get_logger(__name__)


class EyeStateStream(Stream):
    """
    Eye state data

    Each record contains:
        * `ts`: The moment these data were recorded
        * The diameter of each pupil in mm
            * `pupil_diameter_left`
            * `pupil_diameter_right`
        * The position of each eyeball relative to the scene camera, in mm
            * `eyeball_center_left_x`
            * `eyeball_center_left_y`
            * `eyeball_center_left_z`
            * `eyeball_center_right_x`
            * `eyeball_center_right_y`
            * `eyeball_center_right_z`
        * A vector in the forward direction of each eye's optical axis
            * `optical_axis_left_x`
            * `optical_axis_left_y`
            * `optical_axis_left_z`
            * `optical_axis_right_x`
            * `optical_axis_right_y`
            * `optical_axis_right_z`
    """

    def __init__(self, recording):
        log.info("NeonRecording: Loading eye state data")

        eye_state_files = find_sorted_multipart_files(recording._rec_dir, "eye_state")
        eye_state_data, time_data = load_multipart_data_time_pairs(eye_state_files, "<f4", 2)

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

        super().__init__("eye_state", recording, data)
