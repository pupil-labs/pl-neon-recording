import pathlib
import numpy as np

from . import structlog
log = structlog.get_logger(__name__)

def parse_calib_bin(rec_dir: pathlib.Path):
    log.debug("NeonRecording: Loading calibration.bin data")

    calib_raw_data: bytes = b''
    try:
        with open(rec_dir / 'calibration.bin', 'rb') as f:
            calib_raw_data = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {rec_dir / 'calibration.bin'}. Please double check the recording download.")
    except OSError:
        raise OSError(f"Error opening file: {rec_dir / 'calibration.bin'}")
    except Exception as e:
        log.exception(f"Unexpected error loading calibration.bin: {e}")
        raise

    log.debug("NeonRecording: Parsing calibration data")

    # obtained from @dom: 
    # https://github.com/pupil-labs/realtime-python-api/blob/main/src/pupil_labs/realtime_api/device.py#L178
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
