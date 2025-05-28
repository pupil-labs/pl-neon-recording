import logging
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.timeseries.array_record import Array, Record, fields
from pupil_labs.neon_recording.timeseries.timeseries import Timeseries, TimeseriesProps
from pupil_labs.neon_recording.utils import load_multipart_data_time_pairs

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class EventProps(TimeseriesProps):
    event: npt.NDArray[np.str_] = fields[np.str_]("event")  # type:ignore
    "Event name"


class EventRecord(Record, EventProps):
    def keys(self):
        return [x for x in dir(EventProps) if not x.startswith("_") and x != "keys"]


class EventArray(Array[EventRecord], EventProps):
    record_class = EventRecord


class EventTimeseries(Timeseries[EventArray, EventRecord], EventProps):
    """Event annotations"""

    name: str = "event"

    def _load_data_from_recording(self, recording: "NeonRecording") -> EventArray:
        log.debug("NeonRecording: Loading event data")

        events_file = recording._rec_dir / "event.txt"
        time_file = events_file.with_suffix(".time")
        file_pairs = []
        if events_file.exists() and time_file.exists():
            file_pairs = [(events_file, time_file)]
        data = load_multipart_data_time_pairs(file_pairs, "str")
        data.dtype.names = [
            "event" if name == "text" else name for name in data.dtype.names
        ]
        data = data.view(EventArray)
        return data

    @cached_property
    def by_name(self):
        """Return a dict of event_name => all ts"""
        result = defaultdict(list)
        for name, ts in zip(self._data.event, self._data.time, strict=False):
            result[name].append(ts)
        return {key: np.array(ts) for key, ts in result.items()}
