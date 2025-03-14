from pathlib import Path

import numpy as np
import pytest

import pupil_labs.neon_recording as nr


@pytest.fixture
def correct_data(rec_dir: Path) -> np.ndarray:
    calib_path = rec_dir / "calibration.bin"

    dtype = np.dtype(
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

    calibration = np.fromfile(calib_path, dtype)[0]
    return calibration


def test_calibration(rec: nr.NeonRecording, correct_data: np.ndarray):
    assert rec.calibration.version == correct_data["version"]
    assert rec.calibration.serial == correct_data["serial"].decode()

    assert np.all(
        rec.calibration.scene_extrinsics_affine_matrix
        == correct_data["scene_extrinsics_affine_matrix"]
    )
    assert np.all(
        rec.calibration.scene_camera_matrix == correct_data["scene_camera_matrix"]
    )
    assert np.all(
        rec.calibration.scene_distortion_coefficients
        == correct_data["scene_distortion_coefficients"]
    )

    assert np.all(
        rec.calibration.right_extrinsics_affine_matrix
        == correct_data["right_extrinsics_affine_matrix"]
    )

    assert np.all(
        rec.calibration.right_camera_matrix == correct_data["right_camera_matrix"]
    )

    assert np.all(
        rec.calibration.right_distortion_coefficients
        == correct_data["right_distortion_coefficients"]
    )
    assert np.all(
        rec.calibration.left_extrinsics_affine_matrix
        == correct_data["left_extrinsics_affine_matrix"]
    )

    assert np.all(
        rec.calibration.left_camera_matrix == correct_data["left_camera_matrix"]
    )
    assert np.all(
        rec.calibration.left_distortion_coefficients
        == correct_data["left_distortion_coefficients"]
    )
