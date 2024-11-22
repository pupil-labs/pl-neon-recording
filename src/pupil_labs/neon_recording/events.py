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
from pupil_labs.video.array_like import ArrayLike


class EventRecord(NamedTuple):
    ts: int
    event_name: str


class Events(NeonTimeseries[EventRecord]):
    def __init__(self, time_data: ArrayLike[int], event_names: ArrayLike[str]):
        self._time_data = np.array(time_data)
        self._event_names = np.array(event_names)

    @staticmethod
    def from_native_recording(rec_dir: Path) -> "Events":
        events_file = rec_dir / "event.txt"
        time_file = events_file.with_suffix(".time")
        event_names, time_data = load_multipart_data_time_pairs(
            [(events_file, time_file)], "str", 1
        )
        return Events(time_data, event_names)

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._time_data

    ts = timestamps

    @property
    def event_names(self) -> npt.NDArray[np.str_]:
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
                self._time_data[key],
                self._event_names[key],
            )
            return record
        elif isinstance(key, slice):
            return Events(
                self._time_data[key],
                self._event_names[key],
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
