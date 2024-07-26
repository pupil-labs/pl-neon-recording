import numpy as np

from .. import structlog
from ..utils import load_multipart_data_time_pairs
from .stream import Stream

log = structlog.get_logger(__name__)


class EventStream(Stream):
    """
    Event annotations

    Each record contains:
        * `ts`: The moment these data were recorded
        * `event`: The name of the event
    """

    def __init__(self, recording):
        log.info("NeonRecording: Loading event data")

        events_file = recording._rec_dir / "event.txt"
        time_file = events_file.with_suffix(".time")
        if events_file.exists and time_file.exists():
            event_names, time_data = load_multipart_data_time_pairs([(events_file, time_file)], "str", 1)

        data = np.rec.fromarrays(
            [time_data, event_names],
            names=["ts", "event"]
        )

        super().__init__("event", recording, data)

    def unique(self):
        return dict(zip(self.data.event, self.data.ts))
