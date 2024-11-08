from pathlib import Path

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)


class EyeState(dict):
    timestamps: npt.NDArray[np.int64]
    pupil_diameter_left: npt.NDArray[np.float64]
    eyeball_center_left_x: npt.NDArray[np.float64]
    eyeball_center_left_y: npt.NDArray[np.float64]
    eyeball_center_left_z: npt.NDArray[np.float64]
    optical_axis_left_x: npt.NDArray[np.float64]
    optical_axis_left_y: npt.NDArray[np.float64]
    optical_axis_left_z: npt.NDArray[np.float64]
    pupil_diameter_right: npt.NDArray[np.float64]
    eyeball_center_right_x: npt.NDArray[np.float64]
    eyeball_center_right_y: npt.NDArray[np.float64]
    eyeball_center_right_z: npt.NDArray[np.float64]
    optical_axis_right_x: npt.NDArray[np.float64]
    optical_axis_right_y: npt.NDArray[np.float64]
    optical_axis_right_z: npt.NDArray[np.float64]

    def __init__(self, rec_dir: Path):
        eye_state_files = find_sorted_multipart_files(rec_dir, "eye_state")
        eye_state_data, time_data = load_multipart_data_time_pairs(
            eye_state_files, "<f4", 2
        )
        data = eye_state_data.reshape(-1, 14)

        self["timestamps"] = time_data
        self["pupil_diameter_left"] = data[:, 0]
        self["eyeball_center_left_x"] = data[:, 1]
        self["eyeball_center_left_y"] = data[:, 2]
        self["eyeball_center_left_z"] = data[:, 3]
        self["optical_axis_left_x"] = data[:, 4]
        self["optical_axis_left_y"] = data[:, 5]
        self["optical_axis_left_z"] = data[:, 6]
        self["pupil_diameter_right"] = data[:, 7]
        self["eyeball_center_right_x"] = data[:, 8]
        self["eyeball_center_right_y"] = data[:, 9]
        self["eyeball_center_right_z"] = data[:, 10]
        self["optical_axis_right_x"] = data[:, 11]
        self["optical_axis_right_y"] = data[:, 12]
        self["optical_axis_right_z"] = data[:, 13]

    @property
    def pupil_diameters(self) -> npt.NDArray[np.float64]:
        return np.column_stack([self.pupil_diameter_left, self.pupil_diameter_right])

    @property
    def eye_ball_center_left(self) -> npt.NDArray[np.float64]:
        return np.column_stack([
            self.eyeball_center_left_x,
            self.eyeball_center_left_y,
            self.eyeball_center_left_z,
        ])

    @property
    def eye_ball_center_right(self) -> npt.NDArray[np.float64]:
        return np.column_stack([
            self.eyeball_center_right_x,
            self.eyeball_center_right_y,
            self.eyeball_center_right_z,
        ])

    @property
    def optical_axis_left(self) -> npt.NDArray[np.float64]:
        return np.column_stack([
            self.optical_axis_left_x,
            self.optical_axis_left_y,
            self.optical_axis_left_z,
        ])

    @property
    def optical_axis_right(self) -> npt.NDArray[np.float64]:
        return np.column_stack([
            self.optical_axis_right_x,
            self.optical_axis_right_y,
            self.optical_axis_right_z,
        ])

    def __getattr__(self, key: str) -> npt.NDArray:
        return self[key]
