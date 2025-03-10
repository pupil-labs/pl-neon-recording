import logging

from ..utils import load_multipart_data_time_pairs
from .stream import Stream

log = logging.getLogger(__name__)


class EventStream(Stream):
    """
    Event annotations

    Each record contains:
        * `ts`: The moment these data were recorded
        * `event`: The name of the event
    """

    def __init__(self, recording):
        log.debug("NeonRecording: Loading event data")

        events_file = recording._rec_dir / "event.txt"
        time_file = events_file.with_suffix(".time")
        file_pairs = []
        if events_file.exists and time_file.exists():
            file_pairs = [(events_file, time_file)]
        data = load_multipart_data_time_pairs(file_pairs, "str")
        data.dtype.names = [
            "event" if name == "text" else name for name in data.dtype.names
        ]
        super().__init__("event", recording, data)

    def unique(self):
        return dict(zip(self.data.event, self.data.ts))
