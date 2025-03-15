"""Camera calibration utils"""

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.stream.array_record import Record, fields


class Calibration(Record):
    """Camera Calibration data"""

    dtype: np.dtype = np.dtype(
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
    serial = fields[str]("serial", bytes.decode)
    scene_camera_matrix = fields[npt.NDArray[np.float64]]("scene_camera_matrix")
    scene_distortion_coefficients = fields[npt.NDArray[np.float64]](
        "scene_distortion_coefficients"
    )
    scene_extrinsics_affine_matrix = fields[npt.NDArray[np.float64]](
        "scene_extrinsics_affine_matrix"
    )
    right_camera_matrix = fields[npt.NDArray[np.float64]]("right_camera_matrix")
    right_distortion_coefficients = fields[npt.NDArray[np.float64]](
        "right_distortion_coefficients"
    )
    right_extrinsics_affine_matrix = fields[npt.NDArray[np.float64]](
        "right_extrinsics_affine_matrix"
    )
    left_camera_matrix = fields[npt.NDArray[np.float64]]("left_camera_matrix")
    left_distortion_coefficients = fields[npt.NDArray[np.float64]](
        "left_distortion_coefficients"
    )
    left_extrinsics_affine_matrix = fields[npt.NDArray[np.float64]](
        "left_extrinsics_affine_matrix"
    )
    crc = fields[int]("crc")

    @classmethod
    def from_buffer(cls, buffer: bytes):
        return cls(buffer)

    @classmethod
    def from_file(cls, path: str) -> "Calibration":
        return cls(path)
