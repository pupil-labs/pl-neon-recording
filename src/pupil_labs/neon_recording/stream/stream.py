from typing import TYPE_CHECKING, Generic, Literal, SupportsIndex, TypeVar, overload

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.constants import TIMESTAMP_FIELD_NAME
from pupil_labs.neon_recording.stream.array_record import Array, fields

if TYPE_CHECKING:
    from pupil_labs.neon_recording.neon_recording import NeonRecording

RecordType = TypeVar("RecordType")
MatchMethod = Literal["nearest", "before"]


def _record_truthiness(self: np.record):
    if not self.dtype.names:
        return bool(self)
    return all(not np.isnan(self[field]) for field in self.dtype.names)


np.record.__bool__ = _record_truthiness  # type: ignore


class SimpleDataSampler:
    _data: Array
    _ts: npt.NDArray[np.int64]
    ts: npt.NDArray[np.int64]

    def __init__(self, data):
        self._data = data

    def sample(self, tstamps=None, method: MatchMethod = "nearest"):
        if tstamps is None:
            tstamps = self.ts

        if np.ndim(tstamps) == 0:
            tstamps = [tstamps]

        tstamps = np.array(tstamps).astype(np.int64)
        sampler = {
            "nearest": self._sample_nearest,
            "before": self._sample_nearest_before,
        }[method]
        return sampler(tstamps)

    def _sample_nearest(self, ts):
        # Use searchsorted to get the insertion points
        idxs = np.searchsorted(self.ts, ts)

        # Ensure index bounds are valid
        idxs = np.clip(idxs, 1, len(self.ts) - 1)
        left = self.ts[idxs - 1]
        right = self.ts[idxs]

        # Determine whether the left or right value is closer
        idxs -= (np.abs(ts - left) < np.abs(ts - right)).astype(int)

        return self._data[idxs]

    def _sample_nearest_before(self, ts):
        last_idx = len(self._data) - 1
        idxs = np.searchsorted(self.ts, ts)
        idxs[idxs > last_idx] = last_idx

        return self._data[idxs]

    def __iter__(self):
        yield from self.data

    def __len__(self):
        return len(self.ts)

    def to_numpy(self):
        return self._data

    @property
    def data(self):
        return self._data

    def interpolate(self, sorted_ts: npt.NDArray[np.int64]):
        """Interpolated stream data for `sorted_ts`"""
        sorted_ts = np.array(sorted_ts)

        interpolated_dtype = np.dtype([
            (k, np.int64 if k == TIMESTAMP_FIELD_NAME else np.float64)
            for k in self.data.dtype.names
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
        return result.view(self.data.__class__)


class StreamProps:
    ts: npt.NDArray[np.int64] = fields[np.int64](TIMESTAMP_FIELD_NAME)  # type:ignore
    "The moment these data were recorded"

    def keys(self):
        return dir(self)


class Stream(SimpleDataSampler, StreamProps, Generic[RecordType]):
    _data: Array

    def __init__(self, name, recording: "NeonRecording", data):
        self.name = name
        self.recording = recording
        self._data = data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, recording={self.recording!r}, data={self._data!r})"
        )

    @overload
    def __getitem__(self, key: SupportsIndex) -> RecordType: ...
    @overload
    def __getitem__(self, key: slice | str) -> Array: ...
    def __getitem__(self, key: SupportsIndex | slice | str) -> Array | RecordType:
        return self._data[key]

    def __getattr__(self, key):
        return getattr(self._data, key)

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        if not self._data.dtype:
            return ["data"]
        return self._data.dtype.names

    @property
    def pd(self):
        """Return stream data as a pandas DataFrame"""
        return self.data.pd
