from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.stream.array_record import (
    Array,
    Record,
    proxy,
)
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

from .. import structlog
from .stream import Stream, StreamProps

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class BlinkProps(StreamProps):
    start_ts = proxy[np.int64]("blink_start_ts_ns")
    "Start timestamp of blink"

    end_ts = proxy[np.int64]("blink_end_ts_ns")
    "End timestamp of blink"


class BlinkRecord(Record, BlinkProps):
    def keys(self):
        return [x for x in BlinkProps.__dict__.keys() if not x.startswith("_")]


class BlinkArray(Array[BlinkRecord], BlinkProps):
    record_class = BlinkRecord


class BlinkStream(Stream[BlinkRecord], BlinkProps):
    """
    Blink data
    """

    data: BlinkArray

    def __init__(self, recording: "NeonRecording"):
        log.debug("NeonRecording: Loading blink data")
        file_pairs = find_sorted_multipart_files(recording._rec_dir, "blinks")
        data = load_multipart_data_time_pairs(
            file_pairs,
            np.dtype(
                [
                    ("blink_start_ts_ns", "int64"),
                    ("blink_end_ts_ns", "int64"),
                ]
            ),
        )
        super().__init__("blink", recording, data.view(BlinkArray))
