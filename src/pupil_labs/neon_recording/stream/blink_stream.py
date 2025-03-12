import logging
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.constants import (
    DTYPE_END_TIMESTAMP_FIELD_NAME,
    DTYPE_START_TIMESTAMP_FIELD_NAME,
)
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
    start_ts = fields[np.int64](DTYPE_START_TIMESTAMP_FIELD_NAME)
    "Start timestamp of blink"

    end_ts = fields[np.int64](DTYPE_END_TIMESTAMP_FIELD_NAME)
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
            np.dtype([
                (DTYPE_START_TIMESTAMP_FIELD_NAME, "int64"),
                (DTYPE_END_TIMESTAMP_FIELD_NAME, "int64"),
            ]),
        )
        super().__init__("blink", recording, data.view(BlinkArray))
