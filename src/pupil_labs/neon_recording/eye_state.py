from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)


class EyeStateRecord(NamedTuple):
    ts: int
    pupil_diameter_left: float
    eyeball_center_left_x: float
    eyeball_center_left_y: float
    eyeball_center_left_z: float
    optical_axis_left_x: float
    optical_axis_left_y: float
    optical_axis_left_z: float
    pupil_diameter_right: float
    eyeball_center_right_x: float
    eyeball_center_right_y: float
    eyeball_center_right_z: float
    optical_axis_right_x: float
    optical_axis_right_y: float
    optical_axis_right_z: float


class EyeState:
    def __init__(self, rec_dir: Path):
        eye_state_files = find_sorted_multipart_files(rec_dir, "eye_state")
        eye_state_data, time_data = load_multipart_data_time_pairs(
            eye_state_files, "<f4", 2
        )
        self._time_data = time_data
        self._data = eye_state_data.reshape(-1, 14)

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._time_data

    ts = timestamps

    @property
    def pupil_diameters(self) -> npt.NDArray[np.float64]:
        return self._data[:, [0, 7]]

    @property
    def eye_ball_center_left(self) -> npt.NDArray[np.float64]:
        return self._data[:, [1, 2, 3]]

    @property
    def eye_ball_center_right(self) -> npt.NDArray[np.float64]:
        return self._data[:, [8, 9, 10]]

    @property
    def optical_axis_left(self) -> npt.NDArray[np.float64]:
        return self._data[:, [4, 5, 6]]

    @property
    def optical_axis_right(self) -> npt.NDArray[np.float64]:
        return self._data[:, [11, 12, 13]]

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return self._data

    def __getitem__(self, key: int) -> EyeStateRecord:
        record = EyeStateRecord(self._time_data[key], *self._data[key])
        return record
