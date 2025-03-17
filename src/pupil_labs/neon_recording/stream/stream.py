from typing import (
    TYPE_CHECKING,
    Generic,
    Iterator,
    Literal,
    Self,
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.neon_recording.constants import TIMESTAMP_FIELD_NAME
from pupil_labs.neon_recording.stream.array_record import Array, Record, fields

if TYPE_CHECKING:
    from pupil_labs.neon_recording.neon_recording import NeonRecording

ArrayType = TypeVar("ArrayType", bound=Array)
RecordType = TypeVar("RecordType", bound=Record)

MatchMethod = Literal["nearest", "before", "after"]

MATCHING_METHOD_TO_PANDAS_DIRECTION: dict[
    MatchMethod, Literal["backward", "forward", "nearest"]
] = {
    "nearest": "nearest",
    "before": "backward",
    "after": "forward",
}


def _record_truthiness(self: np.record):
    if not self.dtype.names:
        return bool(self)
    return all(not np.isnan(self[field]) for field in self.dtype.names)


np.record.__bool__ = _record_truthiness  # type: ignore


class StreamProps:
    ts: npt.NDArray[np.int64] = fields[np.int64](TIMESTAMP_FIELD_NAME)  # type:ignore
    "The moment these data were recorded"

    def keys(self):
        return dir(self)

    def __len__(self):
        return len(self.keys())


StreamType = TypeVar("StreamType", bound="Stream")


class SampledStream(
    # dict,  # this is so pandas.DataFrame can be called on the stream directly
    StreamProps,
    Generic[ArrayType, RecordType],
):
    def __init__(self, data):
        self._data = data

    @overload
    def __getitem__(self, key: SupportsIndex) -> RecordType: ...
    @overload
    def __getitem__(self, key: slice | str | list[str]) -> ArrayType: ...
    def __getitem__(
        self, key: SupportsIndex | slice | str | list[str]
    ) -> ArrayType | RecordType:
        return self._data[key]  # type: ignore

    def __iter__(self: "Stream") -> Iterator[RecordType]:
        for i in range(len(self)):
            yield self.data[i]

    def keys(self):
        if not self._data.dtype:
            return ["0"]
        return self._data.dtype.names

    def sample(
        self,
        start_or_timestamps: int | np.integer | None = None,
        stop: int | np.integer | None = None,
        tolerance: int | None = None,
        method: MatchMethod = "nearest",
    ) -> Self:
        if start_or_timestamps is None:
            return self

        if method not in MATCHING_METHOD_TO_PANDAS_DIRECTION:
            raise ValueError(
                f"invalid matching method: {method!r}, must be one of: "
                f"{' | '.join(MATCHING_METHOD_TO_PANDAS_DIRECTION.keys())}"
            )

        if isinstance(start_or_timestamps, (int, np.integer, np.floating, float)):
            # TODO: implement before/after
            start = start_or_timestamps
            if stop is not None:
                start_idx, stop_idx = np.searchsorted(self.ts, np.array([start, stop]))
                return self._data[start_idx:stop_idx]
            timestamps = np.array([start])
        else:
            timestamps = start_or_timestamps
            if isinstance(timestamps, set):
                timestamps = sorted(timestamps)
            if sorted(timestamps) != list(timestamps):
                warn(
                    (
                        f"{self.__class__.sample.__qualname__} was called with unsorted"
                        " timestamps, which can cause slower iteration for some streams"
                    ),
                    stacklevel=1,
                )

        if not len(self):
            return self._data[0:0]

        direction = MATCHING_METHOD_TO_PANDAS_DIRECTION[method]

        target_ts = np.array(timestamps).astype(np.int64)
        target_df = pd.DataFrame(target_ts, columns=["target_ts"])
        target_df.index.name = "target"
        target_df.reset_index(inplace=True)  # noqa: PD002

        data_df = pd.DataFrame(self.ts.astype(np.int64), columns=["data_ts"])
        data_df.index.name = "data"
        data_df.reset_index(inplace=True)  # noqa: PD002

        matching_df = pd.merge_asof(
            target_df,
            data_df,
            left_on="target_ts",
            right_on="data_ts",
            direction=direction,
            tolerance=None if tolerance is None else int(tolerance),
        )
        closest_indices = matching_df["data"]
        idxs = closest_indices[closest_indices.notna()].to_numpy().astype(np.int_)
        return SampledStream(self._data[idxs])  # type: ignore

    def __len__(self):
        return len(self.ts)

    def __array__(self):
        return np.array(self._data)  # , dtype=dtype)

    @property
    def data(self) -> Array:
        """Stream data as structured numpy array"""
        return self._data

    @property
    def pd(self):
        """Stream data as a pandas DataFrame"""
        return self.data.pd

    def interpolate(self, sorted_ts: npt.NDArray[np.int64]) -> ArrayType:
        """Interpolated stream data for `sorted_ts`"""
        assert self.data.dtype is not None

        sorted_ts = np.array(sorted_ts)
        interpolated_dtype = np.dtype([
            (k, np.int64 if k == TIMESTAMP_FIELD_NAME else np.float64)
            for k in self.data.dtype.names or []
            if issubclass(self.data.dtype[k].type, (np.floating, np.integer))
        ])

        result = np.zeros(len(sorted_ts), interpolated_dtype)
        result[TIMESTAMP_FIELD_NAME] = sorted_ts
        for key in interpolated_dtype.names or []:
            if key == TIMESTAMP_FIELD_NAME:
                continue
            value = self.data[key].astype(np.float64)
            result[key] = np.interp(
                sorted_ts,
                self.ts,
                value,
                left=np.nan,
                right=np.nan,
            )
        return cast(ArrayType, result.view(self.data.__class__))

    def __repr__(self):
        return f"{self.__class__.__name__}" f"(data={self.data!r})"


class Stream(SampledStream[ArrayType, RecordType]):
    _data: ArrayType
    name: str

    def __init__(self, name, recording: "NeonRecording", data):
        self.name = name
        self.recording = recording
        self._data = data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, recording={self.recording!r}, data={self._data!r})"
        )
