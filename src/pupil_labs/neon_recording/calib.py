"""Camera calibration utils"""

import typing as T

import numpy as np
import numpy.typing as npt


class Calibration(T.NamedTuple):
    """Camera Calibration data"""

    dtype = np.dtype(
        [
            ("version", "u1"),
            ("serial", "6a"),
            ("scene_camera_matrix", "(3,3)d"),
            ("scene_distortion_coefficients", "8d"),
            ("scene_extrinsics_affine_matrix", "(4,4)d"),
            ("right_camera_matrix", "(3,3)d"),
            ("right_distortion_coefficients", "8d"),
            ("right_extrinsics_affine_matrix", "(4,4)d"),
            ("left_camera_matrix", "(3,3)d"),
            ("left_distortion_coefficients", "8d"),
            ("left_extrinsics_affine_matrix", "(4,4)d"),
            ("crc", "u4"),
        ],
    )

    version: int
    serial: str
    scene_camera_matrix: npt.NDArray[np.float64]
    scene_distortion_coefficients: npt.NDArray[np.float64]
    scene_extrinsics_affine_matrix: npt.NDArray[np.float64]
    right_camera_matrix: npt.NDArray[np.float64]
    right_distortion_coefficients: npt.NDArray[np.float64]
    right_extrinsics_affine_matrix: npt.NDArray[np.float64]
    left_camera_matrix: npt.NDArray[np.float64]
    left_distortion_coefficients: npt.NDArray[np.float64]
    left_extrinsics_affine_matrix: npt.NDArray[np.float64]
    crc: int

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return self[key]

    @classmethod
    def from_buffer(cls, buffer: bytes):
        return cls(*np.frombuffer(buffer, cls)[0])

    @classmethod
    def from_file(cls, path: str) -> "Calibration":
        return cls(*np.fromfile(path, cls)[0])
