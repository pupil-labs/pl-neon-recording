import numpy as np

from .. import structlog
from ..utils import find_sorted_multipart_files, load_multipart_data_time_pairs
from .stream import Stream

log = structlog.get_logger(__name__)


class EyeLidStream(Stream):
    """EyeLid state data

    Each record contains:
        * `ts`: The moment these data were recorded
        * `eyelid_angle_top_left`: The angle of the top left eyelid in degrees
        * `eyelid_angle_bottom_left`: The angle of the bottom left eyelid in degrees
        * `eyelid_aperture_left`: The aperture of the eyelid in mm
    """

    def __init__(self, recording):
        log.debug("NeonRecording: Loading eye state data")

        eye_state_files = find_sorted_multipart_files(recording._rec_dir, "eye_state")
        eye_state_data, time_data = load_multipart_data_time_pairs(
            eye_state_files, "<f4", 2
        )

        if eye_state_data.size % 20 == 0:
            data = eye_state_data.reshape(-1, 20)
        else:
            raise ValueError("This recording does not contain eyelid data.")
        data = np.vstack([time_data, data.T[14:]])
        data = np.rec.fromarrays(
            data,
            names=[
                "ts",
                "eyelid_angle_top_left",
                "eyelid_angle_bottom_left",
                "eyelid_aperture_left",
                "eyelid_angle_top_right",
                "eyelid_angle_bottom_right",
                "eyelid_aperture_right",
            ],
        )

        super().__init__("eyelid", recording, data)
