from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.constants import TIMESTAMP_DTYPE
from pupil_labs.neon_recording.stream.array_record import (
    Array,
    Record,
    join_struct_arrays,
    proxy,
)

from .. import structlog
from .stream import Stream, StreamProps

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class EyeStateProps(StreamProps):
    pupil_diameter_left_mm = proxy[np.float32]("pupil_diameter_left_mm")
    "The diameter of the left pupil in mm"

    eyeball_center_left_x = proxy[np.float32]("eyeball_center_left_x")
    "The x position in mm of the left eyeball relative to the scene camera"

    eyeball_center_left_y = proxy[np.float32]("eyeball_center_left_y")
    "The y position in mm of the left eyeball relative to the scene camera"

    eyeball_center_left_z = proxy[np.float32]("eyeball_center_left_z")
    "The z position in mm of the left eyeball relative to the scene camera"

    optical_axis_left_x = proxy[np.float32]("optical_axis_left_x")
    "The x component for forward direction of the left eye's optical axis"

    optical_axis_left_y = proxy[np.float32]("optical_axis_left_y")
    "The y component for forward direction of the left eye's optical axis"

    optical_axis_left_z = proxy[np.float32]("optical_axis_left_z")
    "The z component for forward direction of the left eye's optical axis"

    pupil_diameter_right_mm = proxy[np.float32]("pupil_diameter_right_mm")
    "The diameter of the right pupil in mm"

    eyeball_center_right_x = proxy[np.float32]("eyeball_center_right_x")
    "The x position in mm of the right eyeball relative to the scene camera"

    eyeball_center_right_y = proxy[np.float32]("eyeball_center_right_y")
    "The y position in mm of the right eyeball relative to the scene camera"

    eyeball_center_right_z = proxy[np.float32]("eyeball_center_right_z")
    "The z position in mm of the right eyeball relative to the scene camera"

    optical_axis_right_x = proxy[np.float32]("optical_axis_right_x")
    "The x component for forward direction of the right eye's optical axis"

    optical_axis_right_y = proxy[np.float32]("optical_axis_right_y")
    "The y component for forward direction of the right eye's optical axis"

    optical_axis_right_z = proxy[np.float32]("optical_axis_right_z")
    "The z component for forward direction of the right eye's optical axis"

    eyelid_angle_top_left = proxy[np.float32]("eyelid_angle_top_left")
    "Top left eyelid angle"

    eyelid_angle_bottom_left = proxy[np.float32]("eyelid_angle_bottom_left")
    "Bottom left eyelid angle"

    eyelid_angle_top_right = proxy[np.float32]("eyelid_angle_top_right")
    "Top Right eyelid angle"

    eyelid_angle_bottom_right = proxy[np.float32]("eyelid_angle_bottom_right")
    "Top Right eyelid angle"

    eyelid_aperture_mm_left = proxy[np.float32]("eyelid_aperture_mm_left")
    "Aperture (in mm) of left eyelid"

    eyelid_aperture_mm_right = proxy[np.float32]("eyelid_aperture_mm_right")
    "Aperture (in mm) of right eyelid"

    # composite attributes
    eyeball_center_left_xyz = proxy[np.float64](
        [
            "eyeball_center_left_x",
            "eyeball_center_left_y",
            "eyeball_center_left_z",
        ]
    )
    "The xyz position in mm of the left eyeball relative to the scene camera"

    pupil_diameter_left_right = proxy[np.float64](
        ["pupil_diameter_left_mm", "pupil_diameter_right_mm"]
    )
    "Pupil Diameter (in mm) for both eyes: (left, right)"

    eyeball_center_left_xyz = proxy[np.float64](
        [
            "eyeball_center_left_x",
            "eyeball_center_left_y",
            "eyeball_center_left_z",
        ]
    )
    "Eyeball center for left eye: (x, y, z)"

    eyeball_center_right_xyz = proxy[np.float64](
        [
            "eyeball_center_right_x",
            "eyeball_center_right_y",
            "eyeball_center_right_z",
        ]
    )
    "The xyz position in mm of the right eyeball relative to the scene camera"

    optical_axis_left_xyz = proxy[np.float64](
        [
            "optical_axis_left_x",
            "optical_axis_left_y",
            "optical_axis_left_z",
        ]
    )
    "A xyz vector in the forward direction of the left eye's optical axis"

    optical_axis_right_xyz = proxy[np.float64](
        [
            "optical_axis_right_x",
            "optical_axis_right_y",
            "optical_axis_right_z",
        ]
    )
    "A xyz vector in the forward direction of the right eye's optical axis"

    eyelid_angle = proxy[np.float64](
        [
            "eyelid_angle_top_left",
            "eyelid_angle_bottom_left",
            "eyelid_angle_top_right",
            "eyelid_angle_bottom_right",
        ]
    )
    "Eyelid angle: (top_left, bottom_left, top_right, bottom_right)"

    eyelid_aperture_left_right_mm = proxy[np.float64](
        [
            "eyelid_aperture_mm_left",
            "eyelid_aperture_mm_right",
        ]
    )
    "Eyelid aperture in mm: (left, right)"


class EyeStateRecord(Record, EyeStateProps): ...


class EyeStateArray(Array[EyeStateRecord], EyeStateProps):
    record_class = EyeStateRecord


class EyeStateStream(Stream[EyeStateRecord], EyeStateProps):
    """
    Eye state data
    """

    data: EyeStateArray

    def __init__(self, recording: "NeonRecording"):
        log.debug("NeonRecording: Loading eye state data")

        time_data = Array(
            recording._rec_dir.glob("eye_state ps*.time"), TIMESTAMP_DTYPE
        )
        eye_state_data = Array(
            recording._rec_dir.glob("eye_state ps*.raw"),
            fallback_dtype=np.dtype(
                [
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
                ]
            ),
        )
        data = join_struct_arrays([time_data, eye_state_data]).view(EyeStateArray)
        super().__init__("eye_state", recording, data)
