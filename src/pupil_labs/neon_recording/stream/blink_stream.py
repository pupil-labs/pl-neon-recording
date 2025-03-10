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


class BlinkProps(StreamProps):
    blink_start_ts_ns = proxy[np.int64]("blink_start_ts_ns")
    "Blink start time"

    blink_end_ts_ns = proxy[np.int64]("blink_end_ts_ns")
    "Blink end time"


class BlinkRecord(Record, BlinkProps):
    def keys(self):
        return [x for x in BlinkProps.__dict__.keys() if not x.startswith("_")]


class BlinkArray(Array[BlinkRecord], BlinkProps):
    record_class = BlinkRecord


class BlinkStream(Stream):
    """
        Blinks
    """

    def __init__(self, recording):
        log.debug("NeonRecording: Loading blink data")

        blink_file_pairs = find_sorted_multipart_files(recording._rec_dir, "blinks")
        time_data = Array([file for _, file in blink_file_pairs], TIMESTAMP_DTYPE)

        blink_data = Array(
            [file for file, _ in blink_file_pairs],
            fallback_dtype=np.dtype(np.dtype([
                ("event_type", "int32"),
                ("blink_start_ts_ns", "int64"),
                ("blink_end_ts_ns", "int64"),
            ]))
        )

        data = join_struct_arrays([time_data, blink_data]).view(BlinkArray)

        super().__init__("blinks", recording, data)
