from logging import getLogger
from pathlib import Path

import numpy as np

from pupil_labs.matching import Sensor

from ..utils import load_multipart_data_time_pairs

log = getLogger(__name__)


class Event(np.ndarray, Sensor):
    def __new__(cls, rec_dir: Path):
        log.debug("NeonRecording: Loading event data")

        events_file = rec_dir / "event.txt"
        time_file = events_file.with_suffix(".time")
        if events_file.exists and time_file.exists():
            event_names, time_data = load_multipart_data_time_pairs(
                [(events_file, time_file)], "str", 1
            )

        data = np.rec.fromarrays([time_data, event_names], names=["ts", "event"])
        data = data.view(cls)

        data.timestamps = data["ts"]
        return data
