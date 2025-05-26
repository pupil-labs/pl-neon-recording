import logging
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.timeseries.array_record import (
    Array,
    Record,
    fields,
)
from pupil_labs.neon_recording.timeseries.timeseries import Timeseries, TimeseriesProps
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class BlinkProps(TimeseriesProps):
    start_ts = fields[np.int64]("start_timestamp_ns")  # type:ignore
    "Start timestamp of blink"

    end_ts = fields[np.int64]("end_timestamp_ns")  # type:ignore
    "End timestamp of blink"


class BlinkRecord(Record, BlinkProps):
    def keys(self):
        return [x for x in dir(BlinkProps) if not x.startswith("_") and x != "keys"]


class BlinkArray(Array[BlinkRecord], BlinkProps):
    record_class = BlinkRecord


class BlinkTimeseries(Timeseries[BlinkArray, BlinkRecord], BlinkProps):
    """Blinks data"""

    name: str = "blink"

    def from_recording(self, recording: "NeonRecording") -> BlinkArray:
        log.debug("NeonRecording: Loading blink data")
        file_pairs = find_sorted_multipart_files(recording._rec_dir, "blinks")
        data = load_multipart_data_time_pairs(
            file_pairs,
            np.dtype([
                ("start_timestamp_ns", "int64"),
                ("end_timestamp_ns", "int64"),
            ]),
        )
        data = data.view(BlinkArray)
        return data
