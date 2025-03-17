import sys
from collections.abc import Iterator
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.constants import TIMESTAMP_FIELD_NAME
from pupil_labs.neon_recording.stream.array_record import Array, Record, fields

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from pupil_labs.neon_recording.neon_recording import NeonRecording

ArrayType = TypeVar("ArrayType", bound=Array)
RecordType = TypeVar("RecordType", bound=Record)
MatchMethod = Literal["nearest", "before"]


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
    _data: ArrayType

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

    def __iter__(self) -> Iterator[RecordType]:
        for i in range(len(self)):
            yield self.data[i]

    def keys(self):
        if not self._data.dtype:
            return ["0"]
        return self._data.dtype.names

    def sample(self, tstamps=None, method: MatchMethod = "nearest") -> Self:
        if tstamps is None:
            tstamps = self.ts

        if np.ndim(tstamps) == 0:
            tstamps = [tstamps]

        tstamps = np.array(tstamps).astype(np.int64)
        sampler = {
            "nearest": self._sample_nearest,
            "before": self._sample_nearest_before,
        }[method]
        idxs = sampler(tstamps)

        return SampledStream(self._data[idxs])  # type: ignore

    def _sample_nearest(self, ts):
        # Use searchsorted to get the insertion points
        idxs = np.searchsorted(self.ts, ts)

        # Ensure index bounds are valid
        idxs = np.clip(idxs, 1, len(self.ts) - 1)
        left = self.ts[idxs - 1]
        right = self.ts[idxs]

        # Determine whether the left or right value is closer
        idxs -= (np.abs(ts - left) < np.abs(ts - right)).astype(int)

        return idxs

    def _sample_nearest_before(self, ts):
        last_idx = len(self._data) - 1
        idxs = np.searchsorted(self.ts, ts)
        idxs[idxs > last_idx] = last_idx

        return idxs

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
        return f"{self.__class__.__name__}(data={self.data!r})"


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
