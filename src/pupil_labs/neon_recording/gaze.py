from pathlib import Path
from typing import Iterator, NamedTuple, Optional, overload

import numpy as np
import numpy.typing as npt

from pupil_labs.matching import MatchedIndividual, MatchingMethod
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)
from pupil_labs.video.array_like import ArrayLike


class GazeRecord(NamedTuple):
    ts: int
    x: float
    y: float

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y


class Gaze(ArrayLike[GazeRecord]):
    def __init__(self, time_data: ArrayLike[int], gaze_data: ArrayLike[float]):
        assert len(time_data) == len(gaze_data)
        self._time_data = np.array(time_data)
        self._gaze_data = np.array(gaze_data)

    @staticmethod
    def from_native_recording(rec_dir: Path):
        gaze_200hz_file = rec_dir / "gaze_200hz.raw"
        time_200hz_file = rec_dir / "gaze_200hz.time"
        gaze_file_pairs = []
        if gaze_200hz_file.exists() and time_200hz_file.exists():
            gaze_file_pairs.append((gaze_200hz_file, time_200hz_file))
        else:
            gaze_file_pairs = find_sorted_multipart_files(rec_dir, "gaze")
        gaze_data, time_data = load_multipart_data_time_pairs(gaze_file_pairs, "<f4", 2)

        return Gaze(time_data, gaze_data)

    @property
    def timestamps(self) -> npt.NDArray[np.float64]:
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

    def __len__(self) -> int:
        return len(self._time_data)

    @overload
    def __getitem__(self, key: int, /) -> GazeRecord: ...
    @overload
    def __getitem__(self, key: slice, /) -> "Gaze": ...
    def __getitem__(self, key: int | slice) -> "GazeRecord | Gaze":
        if isinstance(key, int):
            record = GazeRecord(self._time_data[key], *self._gaze_data[key])
        elif isinstance(key, slice):
            return Gaze(self._time_data[key], self._gaze_data[key])
        else:
            raise TypeError(f"Invalid argument type {type(key)}")
        return record

    def __iter__(self) -> Iterator[GazeRecord]:
        for i in range(len(self)):
            yield self[i]

    def sample(
        self,
        timestamps: ArrayLike[int] | ArrayLike[float],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[float] = None,
    ) -> MatchedIndividual:
        return MatchedIndividual(
            timestamps,
            self,
            method=method,
            tolerance=tolerance,
        )


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
