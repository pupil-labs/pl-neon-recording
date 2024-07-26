import pathlib
from dataclasses import dataclass

import numpy as np

from . import structlog

log = structlog.get_logger(__name__)


@dataclass
class Calibration:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    extrinsics_affine_matrix: np.ndarray


def _parse_calib_bin(rec_dir: pathlib.Path):
    log.debug("NeonRecording: Loading calibration.bin data")

    calib_raw_data: bytes = b""
    try:
        with open(rec_dir / "calibration.bin", "rb") as f:
            calib_raw_data = f.read()
    except Exception as e:
        log.exception(f"Unexpected error loading calibration.bin: {e}")
        raise

    log.debug("NeonRecording: Parsing calibration data")

    return np.frombuffer(
        calib_raw_data,
        np.dtype(
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
            ]
        ),
    )
