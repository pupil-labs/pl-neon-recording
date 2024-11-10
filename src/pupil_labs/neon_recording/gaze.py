from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)


class GazeRecord(NamedTuple):
    ts: int
    x: float
    y: float


class Gaze:
    def __init__(self, rec_dir: Path):
        gaze_200hz_file = rec_dir / "gaze_200hz.raw"
        time_200hz_file = rec_dir / "gaze_200hz.time"
        gaze_file_pairs = []
        if gaze_200hz_file.exists() and time_200hz_file.exists():
            gaze_file_pairs.append((gaze_200hz_file, time_200hz_file))
        else:
            gaze_file_pairs = find_sorted_multipart_files(rec_dir, "gaze")
        gaze_data, time_data = load_multipart_data_time_pairs(gaze_file_pairs, "<f4", 2)
        self._time_data = np.array(time_data)
        self._gaze_data = np.array(gaze_data)

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._time_data

    ts = timestamps

    @property
    def xy(self) -> npt.NDArray[np.float64]:
        return self._gaze_data

    @property
    def x(self) -> npt.NDArray[np.float64]:
        return self._gaze_data[:, 0]

    @property
    def y(self) -> npt.NDArray[np.float64]:
        return self._gaze_data[:, 1]

    def __getitem__(self, key: int) -> GazeRecord:
        record = GazeRecord(self._time_data[key], *self._gaze_data[key])
        return record


# The issue with the below is that it doesn't suppored mixed data types
class Gaze2(npt.NDArray[np.float64]):
    def __new__(cls, rec_dir: Path):
        gaze_200hz_file = rec_dir / "gaze_200hz.raw"
        time_200hz_file = rec_dir / "gaze_200hz.time"
        gaze_file_pairs = []
        if gaze_200hz_file.exists() and time_200hz_file.exists():
            gaze_file_pairs.append((gaze_200hz_file, time_200hz_file))
        else:
            gaze_file_pairs = find_sorted_multipart_files(rec_dir, "gaze")
        gaze_data, time_data = load_multipart_data_time_pairs(gaze_file_pairs, "<f4", 2)
        data = np.vstack([time_data, gaze_data.T]).T
        return data.view(cls)

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._index_data(0)

    @property
    def x(self) -> npt.NDArray[np.float64]:
        return self._index_data(1)

    @property
    def y(self) -> npt.NDArray[np.float64]:
        return self._index_data(2)

    @property
    def xy(self) -> npt.NDArray[np.float64]:
        return self._index_data(slice(1, 3))

    def _index_data(self, key: int | slice) -> npt.NDArray:
        if len(self.shape) == 1:
            return self[key]
        else:
            return self[:, key]
