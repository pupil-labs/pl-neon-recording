from logging import getLogger
from pathlib import Path

from ..utils import find_sorted_multipart_files, load_multipart_data_time_pairs
from .numpy_timeseries import NumpyTimeseries

log = getLogger(__name__)


class EyeState(NumpyTimeseries):
    def __init__(self, rec_dir: Path):
        log.debug("NeonRecording: Loading eye state data")

        eye_state_files = find_sorted_multipart_files(rec_dir, "eye_state")
        eye_state_data, time_data = load_multipart_data_time_pairs(
            eye_state_files, "<f4", 2
        )

        data = eye_state_data.reshape(-1, 14)
        super().__init__(time_data, data)
