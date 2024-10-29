from logging import getLogger
from pathlib import Path

from ..utils import load_multipart_data_time_pairs
from .numpy_timeseries import NumpyTimeseries

log = getLogger(__name__)


class Event(NumpyTimeseries):
    def __init__(self, rec_dir: Path):
        log.debug("NeonRecording: Loading event data")

        events_file = rec_dir / "event.txt"
        time_file = events_file.with_suffix(".time")
        if events_file.exists and time_file.exists():
            event_names, time_data = load_multipart_data_time_pairs(
                [(events_file, time_file)], "str", 1
            )

        super().__init__(time_data, event_names)
