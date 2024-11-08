from pathlib import Path

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.utils import (
    load_multipart_data_time_pairs,
)


class Events(dict):
    timestamps: npt.NDArray[np.int64]
    event_names: npt.NDArray[np.str_]

    def __init__(self, rec_dir: Path):
        events_file = rec_dir / "event.txt"
        time_file = events_file.with_suffix(".time")
        if not events_file.exists or not time_file.exists():
            raise FileNotFoundError("Event files not found")

        event_names, time_data = load_multipart_data_time_pairs(
            [(events_file, time_file)], "str", 1
        )
        self["timestamps"] = time_data
        self["event_names"] = event_names

    @property
    def xy(self) -> npt.NDArray[np.float64]:
        return np.column_stack([self.x, self.y])

    def __getattr__(self, key: str) -> npt.NDArray:
        return self[key]
