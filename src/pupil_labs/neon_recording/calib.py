from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class Calibration(NamedTuple):
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

    @staticmethod
    def from_file(path: str | Path) -> "Calibration":
        data_buffer = b""
        with open(path, "rb") as f:
            data_buffer += f.read()

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
        data = np.frombuffer(data_buffer, dtype)[0]
        serial = data[1].item().decode()
        return Calibration(
            data[0],
            serial,
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9],
            data[10],
            data[11],
        )
