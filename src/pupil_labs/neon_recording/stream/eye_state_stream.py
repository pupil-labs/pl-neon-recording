import numpy as np

from .. import structlog
from .stream import Stream
from ..utils import find_sorted_multipart_files, load_multipart_data_time_pairs

log = structlog.get_logger(__name__)


class EyeStateStream(Stream):
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
                "pupil diameter left",
                "eyeball center left x",
                "eyeball center left y",
                "eyeball center left z",
                "optical axis left x",
                "optical axis left y",
                "optical axis left z",
                "pupil diameter right",
                "eyeball center right x",
                "eyeball center right y",
                "eyeball center right z",
                "optical axis right x",
                "optical axis right y",
                "optical axis right z",
            ],
        )

        super().__init__("eye_state", recording, data)
