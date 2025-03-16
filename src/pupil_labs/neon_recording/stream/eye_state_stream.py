import logging
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.stream.array_record import Array, Record, fields
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

from .stream import Stream, StreamProps

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class EyeStateProps(StreamProps):
    pupil_diameter_left_right_mm = fields[np.float64]([
        "pupil_diameter_left_mm",
        "pupil_diameter_right_mm",
    ])
    "Pupil diameter (in mm) for both eyes: (left, right)"

    pupil_diameter_left_mm = fields[np.float64](["pupil_diameter_left_mm"])
    "Pupil diameter (in mm) for left eye"

    pupil_diameter_right_mm = fields[np.float64](["pupil_diameter_right_mm"])
    "Pupil diameter (in mm) for right eye"

    eyeball_center_left_xyz = fields[np.float64]([
        "eyeball_center_left_x",
        "eyeball_center_left_y",
        "eyeball_center_left_z",
    ])
    "The xyz position in mm of the left eyeball relative to the scene camera"

    eyeball_center_right_xyz = fields[np.float64]([
        "eyeball_center_right_x",
        "eyeball_center_right_y",
        "eyeball_center_right_z",
    ])
    "The xyz position in mm of the right eyeball relative to the scene camera"

    optical_axis_left_xyz = fields[np.float64]([
        "optical_axis_left_x",
        "optical_axis_left_y",
        "optical_axis_left_z",
    ])
    "A xyz vector in the forward direction of the left eye's optical axis"

    optical_axis_right_xyz = fields[np.float64]([
        "optical_axis_right_x",
        "optical_axis_right_y",
        "optical_axis_right_z",
    ])
    "A xyz vector in the forward direction of the right eye's optical axis"

    eyelid_angle = fields[np.float64]([
        "eyelid_angle_top_left",
        "eyelid_angle_bottom_left",
        "eyelid_angle_top_right",
        "eyelid_angle_bottom_right",
    ])
    "Eyelid angle: (top_left, bottom_left, top_right, bottom_right)"

    eyelid_aperture_left_right_mm = fields[np.float64]([
        "eyelid_aperture_left_mm",
        "eyelid_aperture_right_mm",
    ])
    "Eyelid aperture in mm: (left, right)"


class EyeStateRecord(Record, EyeStateProps):
    def keys(self):
        keys = EyeStateProps.__dict__.keys()
        return [x for x in keys if not x.startswith("_")]


class EyeStateArray(Array[EyeStateRecord], EyeStateProps):
    record_class = EyeStateRecord


class EyeStateStream(Stream[EyeStateArray, EyeStateRecord], EyeStateProps):
    """Eye state data"""

    data: EyeStateArray

    def __init__(self, recording: "NeonRecording"):
        log.debug("NeonRecording: Loading eye state data")
        file_pairs = find_sorted_multipart_files(recording._rec_dir, "eye_state")
        data = load_multipart_data_time_pairs(
            file_pairs,
            dtype=np.dtype([
                ("pupil_diameter_left_mm", "float32"),
                ("eyeball_center_left_x", "float32"),
                ("eyeball_center_left_y", "float32"),
                ("eyeball_center_left_z", "float32"),
                ("optical_axis_left_x", "float32"),
                ("optical_axis_left_y", "float32"),
                ("optical_axis_left_z", "float32"),
                ("pupil_diameter_right_mm", "float32"),
                ("eyeball_center_right_x", "float32"),
                ("eyeball_center_right_y", "float32"),
                ("eyeball_center_right_z", "float32"),
                ("optical_axis_right_x", "float32"),
                ("optical_axis_right_y", "float32"),
                ("optical_axis_right_z", "float32"),
            ]),
        )
        super().__init__("eye_state", recording, data.view(EyeStateArray))
