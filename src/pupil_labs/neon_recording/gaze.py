from functools import cached_property
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.matching import MatchingMethod, SampledData
from pupil_labs.neon_recording.neon_timeseries import NeonTimeseries
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)
from pupil_labs.video import ArrayLike, Indexer


class GazeRecord(NamedTuple):
    timestamp: int
    x: float
    y: float

    @property
    def ts(self) -> int:
        return self.timestamp

    @property
    def xy(self) -> npt.NDArray[np.float64]:
        return np.array([self.x, self.y])

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return np.array([self.ts, self.x, self.y])


class Gaze(NeonTimeseries[GazeRecord]):
    def __init__(
        self, time_data: ArrayLike[int], gaze_data: ArrayLike[float], rec_start: int
    ):
        assert len(time_data) == len(gaze_data)
        self._time_data = np.array(time_data)
        self._gaze_data = np.array(gaze_data)
        self._rec_start = rec_start

    @staticmethod
    def from_native_recording(rec_dir: Path, rec_start: int) -> "Gaze":
        gaze_200hz_file = rec_dir / "gaze_200hz.raw"
        time_200hz_file = rec_dir / "gaze_200hz.time"
        gaze_file_pairs = []
        if gaze_200hz_file.exists() and time_200hz_file.exists():
            gaze_file_pairs.append((gaze_200hz_file, time_200hz_file))
        else:
            gaze_file_pairs = find_sorted_multipart_files(rec_dir, "gaze")
        gaze_data, time_data = load_multipart_data_time_pairs(gaze_file_pairs, "<f4", 2)

        return Gaze(
            time_data,
            gaze_data,
            rec_start,
        )

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._time_data

    ts = timestamps

    @cached_property
    def rel_timestamps(self) -> npt.NDArray[np.float64]:
        """Relative timestamps in seconds in relation to the recording  beginning."""
        return (self.timestamps - self._rec_start) / 1e9

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
            return Gaze(self._time_data[key], self._gaze_data[key], self._rec_start)
        else:
            raise TypeError(f"Invalid argument type {type(key)}")
        return record

    def __iter__(self) -> Iterator[GazeRecord]:
        for i in range(len(self)):
            yield self[i]

    def sample(
        self,
        timestamps: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
    ) -> SampledData[GazeRecord]:
        return SampledData.sample(
            timestamps,
            self,
            method=method,
            tolerance=tolerance,
        )

    def interpolate(self, timestamps: ArrayLike[int]) -> "Gaze":
        timestamps = np.array(timestamps)
        x = np.interp(timestamps, self.timestamps, self.x)
        y = np.interp(timestamps, self.timestamps, self.y)
        xy = np.column_stack((x, y))
        return Gaze(timestamps, xy, self._rec_start)

    @property
    def by_abs_timestamp(self) -> Indexer[GazeRecord]:
        return Indexer(self.timestamps, self)

    @property
    def by_rel_timestamp(self) -> Indexer[GazeRecord]:
        return Indexer(self.rel_timestamps, self)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._gaze_data, columns=["x", "y"], index=self._time_data)
