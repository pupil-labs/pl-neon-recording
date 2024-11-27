from functools import cached_property
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.matching import MatchingMethod, SampledData
from pupil_labs.neon_recording.neon_timeseries import NeonTimeseries
from pupil_labs.neon_recording.utils import (
    load_multipart_data_time_pairs,
)
from pupil_labs.video import ArrayLike, Indexer


class EventRecord(NamedTuple):
    abs_timestamp: int
    rel_timestamp: float
    event_name: str

    @property
    def abs_ts(self) -> int:
        return self.abs_timestamp

    @property
    def rel_ts(self) -> float:
        return self.rel_timestamp


class Events(NeonTimeseries[EventRecord]):
    def __init__(
        self, time_data: ArrayLike[int], event_names: ArrayLike[str], rec_start: int = 0
    ):
        self._time_data = np.array(time_data)
        self._event_names = np.array(event_names)
        self._rec_start = rec_start

    @staticmethod
    def from_native_recording(rec_dir: Path, rec_start: int) -> "Events":
        events_file = rec_dir / "event.txt"
        time_file = events_file.with_suffix(".time")
        event_names, time_data = load_multipart_data_time_pairs(
            [(events_file, time_file)], "str", 1
        )
        return Events(time_data, event_names, rec_start)

    @property
    def abs_timestamp(self) -> npt.NDArray[np.int64]:
        return self._time_data

    abs_ts = abs_timestamp

    @cached_property
    def rel_timestamp(self) -> npt.NDArray[np.float64]:
        return (self.abs_timestamp - self._rec_start) / 1e9

    @property
    def rel_ts(self) -> npt.NDArray[np.float64]:
        return self.rel_timestamp

    @property
    def by_abs_timestamp(self) -> Indexer[EventRecord]:
        return Indexer(self.abs_timestamp, self)

    @property
    def by_rel_timestamp(self) -> Indexer[EventRecord]:
        return Indexer(self.rel_timestamp, self)

    @property
    def event_name(self) -> npt.NDArray[np.str_]:
        """The event names."""
        return self._event_names

    def __len__(self) -> int:
        return len(self._time_data)

    @overload
    def __getitem__(self, key: int, /) -> EventRecord: ...
    @overload
    def __getitem__(self, key: slice, /) -> "Events": ...
    def __getitem__(self, key: int | slice) -> "EventRecord | Events":
        if isinstance(key, int):
            record = EventRecord(
                self.abs_timestamp[key],
                self.rel_timestamp[key],
                self._event_names[key],
            )
            return record
        elif isinstance(key, slice):
            return Events(
                self._time_data[key],
                self._event_names[key],
                self._rec_start,
            )
        else:
            raise TypeError(f"Invalid argument type {type(key)}")

    def __iter__(self) -> Iterator[EventRecord]:
        for i in range(len(self)):
            yield self[i]

    def sample(
        self,
        timestamps: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
    ) -> SampledData[EventRecord]:
        return SampledData.sample(
            timestamps,
            self,
            method=method,
            tolerance=tolerance,
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._event_names,
            columns=[
                "event_name",
            ],
            index=self._time_data,
        )
