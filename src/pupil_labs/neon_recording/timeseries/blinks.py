import logging
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.timeseries.array_record import (
    Array,
    Record,
    fields,
)
from pupil_labs.neon_recording.timeseries.timeseries import Timeseries
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class BlinkProps:
    # Note, BlinkProps do not inherit from TimeseriesProps because they should not
    # have a `time` attribute.
    start_time = fields[np.int64]("start_time")  # type:ignore
    "Start timestamp of the blink."

    stop_time = fields[np.int64]("stop_time")  # type:ignore
    "Stop timestamp of the blink."


class BlinkRecord(Record, BlinkProps):
    def keys(self):
        return [x for x in dir(BlinkProps) if not x.startswith("_") and x != "keys"]


class BlinkArray(Array[BlinkRecord], BlinkProps):
    record_class = BlinkRecord


class BlinkTimeseries(Timeseries[BlinkArray, BlinkRecord], BlinkProps):
    """Blink event data."""

    name: str = "blink"

    def _load_data_from_recording(self, recording: "NeonRecording") -> BlinkArray:
        log.debug("NeonRecording: Loading blink data")
        file_pairs = find_sorted_multipart_files(recording._rec_dir, "blinks")
        data = load_multipart_data_time_pairs(
            file_pairs,
            np.dtype([
                ("start_timestamp_ns", "int64"),
                ("end_timestamp_ns", "int64"),
            ]),
        )
        data = data[["time", "start_timestamp_ns", "end_timestamp_ns"]]
        data.dtype.names = ("time", "start_time", "stop_time")
        data = data.view(BlinkArray)
        return data
