import logging
from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.stream.array_record import Array, Record, fields
from pupil_labs.neon_recording.utils import load_multipart_data_time_pairs

from .stream import Stream, StreamProps

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class EventProps(StreamProps):
    event = fields[np.float64]("event")
    "Event name"


class EventRecord(Record, EventProps):
    def keys(self):
        return [x for x in EventProps.__dict__.keys() if not x.startswith("_")]


class EventArray(Array[EventRecord], EventProps):
    record_class = EventRecord


class EventStream(Stream[EventRecord], EventProps):
    """Event annotations"""

    data: EventArray

    def __init__(self, recording: "NeonRecording"):
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
        super().__init__("event", recording, data.view(EventArray))

    @cached_property
    def by_name(self):
        """Return a dict of event_name => all ts"""
        result = defaultdict(list)
        for name, ts in zip(self.data.event, self.data.ts):
            result[name].append(ts)
        return {key: np.array(ts) for key, ts in result.items()}
