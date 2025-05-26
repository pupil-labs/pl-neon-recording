import logging
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.timeseries.array_record import Array, Record, fields
from pupil_labs.neon_recording.timeseries.timeseries import Timeseries, TimeseriesProps
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class WornProps(TimeseriesProps):
    worn = fields[np.float64]("worn")  # type:ignore
    "Worn"


class WornRecord(Record, WornProps):
    def keys(self):
        return [x for x in dir(WornProps) if not x.startswith("_") and x != "keys"]


class WornArray(Array[WornRecord], WornProps):
    record_class = WornRecord


class WornTimeseries(Timeseries[WornArray, WornRecord], WornProps):
    """Worn (headset on/off) data"""

    name: str = "worn"

    def _load_data_from_recording(self, recording) -> "WornArray":
        log.debug("NeonRecording: Loading worn data")

        worn_200hz_file = recording._rec_dir / "worn_200hz.raw"
        time_200hz_file = recording._rec_dir / "gaze_200hz.time"

        file_pairs = []
        if worn_200hz_file.exists() and time_200hz_file.exists():
            log.debug("NeonRecording: Using 200Hz worn data")
            file_pairs.append((worn_200hz_file, time_200hz_file))
        else:
            log.debug("NeonRecording: Using realtime worn data")
            file_pairs = find_sorted_multipart_files(recording._rec_dir, "worn")

        data = load_multipart_data_time_pairs(file_pairs, np.dtype([("worn", "u1")]))
        data = data.view(WornArray)
        return data
