import numpy as np

from .. import structlog
from .stream import Stream, StreamProps
from ..utils import find_sorted_multipart_files

from pupil_labs.neon_recording.constants import TIMESTAMP_DTYPE
from pupil_labs.neon_recording.stream.array_record import (
    Array,
    Record,
    join_struct_arrays,
    proxy,
)

log = structlog.get_logger(__name__)


class FixationProps(StreamProps):
    start_timestamp_ns = proxy[np.int64]("start_timestamp_ns")
    "Fixation start time"

    end_timestamp_ns = proxy[np.int64]("end_timestamp_ns")
    "Fixation end time"

    start_gaze_x = proxy[np.float32]("start_gaze_x")
    "Horizontal position of the gaze when the fixation started"

    start_gaze_y = proxy[np.float32]("start_gaze_y")
    "Vertical position of the gaze when the fixation started"

    end_gaze_x = proxy[np.float32]("end_gaze_x")
    "Horizontal position of the gaze when the fixation ended"

    end_gaze_y = proxy[np.float32]("end_gaze_y")
    "Vertical position of the gaze when the fixation ended"

    mean_gaze_x = proxy[np.float32]("mean_gaze_x")
    "Mean horizontal position of gazes from the fixation"

    mean_gaze_y = proxy[np.float32]("mean_gaze_y")
    "Mean vertical position of gazes from the fixation"

    amplitude_pixels = proxy[np.float32]("amplitude_pixels")
    "Amplitude of the fixation"

    amplitude_angle_deg = proxy[np.float32]("amplitude_angle_deg")
    "Amplitude angle of the fixation"

    mean_velocity = proxy[np.float32]("mean_velocity")
    "Mean velocity of the fixation"

    max_velocity = proxy[np.float32]("max_velocity")
    "Maximum velocity of the fixation"


class FixationRecord(Record, FixationProps):
    def keys(self):
        return [x for x in FixationProps.__dict__.keys() if not x.startswith("_")]


class FixationArray(Array[FixationRecord], FixationProps):
    record_class = FixationRecord


class FixationStream(Stream):
    """
        Fixations
    """

    def __init__(self, recording):
        log.debug("NeonRecording: Loading fixation data")

        fixation_file_pairs = find_sorted_multipart_files(recording._rec_dir, "fixations")
        time_data = Array([file for _, file in fixation_file_pairs], TIMESTAMP_DTYPE)

        fixation_data = Array(
            [file for file, _ in fixation_file_pairs],
            fallback_dtype=np.dtype(np.dtype([
                ("event_type", "int32"),
                ("start_timestamp_ns", "int64"),
                ("end_timestamp_ns", "int64"),
                ("start_gaze_x", "float32"),
                ("start_gaze_y", "float32"),
                ("end_gaze_x", "float32"),
                ("end_gaze_y", "float32"),
                ("mean_gaze_x", "float32"),
                ("mean_gaze_y", "float32"),
                ("amplitude_pixels", "float32"),
                ("amplitude_angle_deg", "float32"),
                ("mean_velocity", "float32"),
                ("max_velocity", "float32"),
            ]))
        )

        data = join_struct_arrays([time_data, fixation_data]).view(FixationArray)

        super().__init__("fixations", recording, data)


