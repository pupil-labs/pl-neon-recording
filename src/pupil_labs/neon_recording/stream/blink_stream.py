import logging
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.stream.array_record import (
    Array,
    Record,
    fields,
)
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

from .stream import Stream, StreamProps

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class BlinkProps(StreamProps):
    start_ts = fields[np.int64]("start_timestamp_ns")
    "Start timestamp of blink"

    end_ts = fields[np.int64]("end_timestamp_ns")
    "End timestamp of blink"


class BlinkRecord(Record, BlinkProps):
    def keys(self):
        keys = BlinkProps.__dict__.keys()
        return [x for x in keys if not x.startswith("_")]


class BlinkArray(Array[BlinkRecord], BlinkProps):
    record_class = BlinkRecord


class BlinkStream(Stream[BlinkRecord], BlinkProps):
    """Blinks data"""

    data: BlinkArray

    def __init__(self, recording: "NeonRecording"):
        log.debug("NeonRecording: Loading blink data")
        file_pairs = find_sorted_multipart_files(recording._rec_dir, "blinks")
        data = load_multipart_data_time_pairs(
            file_pairs,
            np.dtype([
                ("start_timestamp_ns", "int64"),
                ("end_timestamp_ns", "int64"),
            ]),
        )
        super().__init__("blink", recording, data.view(BlinkArray))
