"""Camera calibration utils"""

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.timeseries.array_record import Record, fields


class Calibration(Record):
    """Camera Calibration data"""

    dtype: np.dtype = np.dtype(
        [
            ("version", "u1"),
            ("serial", "6S"),
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
    "Version of the calibration data."

    serial = fields[str]("serial", bytes.decode)
    "Serial number of the Neon module."

    scene_camera_matrix = fields[npt.NDArray[np.float64]]("scene_camera_matrix")
    "Camera matrix of the scene camera."

    scene_distortion_coefficients = fields[npt.NDArray[np.float64]](
        "scene_distortion_coefficients"
    )
    "Distortion coefficients of the scene camera."

    scene_extrinsics_affine_matrix = fields[npt.NDArray[np.float64]](
        "scene_extrinsics_affine_matrix"
    )
    "Extrinsics affine matrix of the scene camera."

    right_camera_matrix = fields[npt.NDArray[np.float64]]("right_camera_matrix")
    "Camera matrix of the right eye camera."

    right_distortion_coefficients = fields[npt.NDArray[np.float64]](
        "right_distortion_coefficients"
    )
    "Distortion coefficients of the right eye camera."

    right_extrinsics_affine_matrix = fields[npt.NDArray[np.float64]](
        "right_extrinsics_affine_matrix"
    )
    "Extrinsics affine matrix of the right eye camera."

    left_camera_matrix = fields[npt.NDArray[np.float64]]("left_camera_matrix")
    "Camera matrix of the left eye camera."

    left_distortion_coefficients = fields[npt.NDArray[np.float64]](
        "left_distortion_coefficients"
    )
    "Distortion coefficients of the left eye camera."

    left_extrinsics_affine_matrix = fields[npt.NDArray[np.float64]](
        "left_extrinsics_affine_matrix"
    )
    "Extrinsics affine matrix of the left eye camera."

    crc = fields[int]("crc")
    "CRC of the calibration data."

    @classmethod
    def from_buffer(cls, buffer: bytes) -> "Calibration":
        return cls(buffer)

    @classmethod
    def from_file(cls, path: str) -> "Calibration":
        return cls(path)
