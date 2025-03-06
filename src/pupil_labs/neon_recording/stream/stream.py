from enum import Enum
from typing import TYPE_CHECKING, Generic, SupportsIndex, TypeVar, overload

import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.constants import TIMESTAMP_FIELD_NAME
from pupil_labs.neon_recording.stream.array_record import Array, proxy

if TYPE_CHECKING:
    from pupil_labs.neon_recording.neon_recording import NeonRecording

RecordType = TypeVar("RecordType")


class InterpolationMethod(Enum):
    NEAREST = "nearest"
    NEAREST_BEFORE = "nearest_before"
    LINEAR = "linear"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other

        return super().__eq__(other)


def _record_truthiness(self):
    for field in self.dtype.names:
        if np.isnan(self[field]):
            return False

    return True


np.record.__bool__ = _record_truthiness


class SimpleDataSampler:
    _ts: npt.NDArray[np.int64]
    _data: npt.NDArray

    sampler_class = None

    def __init__(self, data):
        self._data = data

    def sample(self, tstamps=None, method=InterpolationMethod.NEAREST):
        if tstamps is None:
            tstamps = self.ts

        if np.ndim(tstamps) == 0:
            tstamps = [tstamps]

        tstamps = np.array(tstamps).astype(np.int64)
        if method == InterpolationMethod.NEAREST:
            return self._sample_nearest(tstamps)

        if method == InterpolationMethod.NEAREST_BEFORE:
            return self._sample_nearest_before(tstamps)

        elif method == InterpolationMethod.LINEAR:
            return self._sample_linear_interp(tstamps)

    def _sample_nearest(self, ts):
        # Use searchsorted to get the insertion points
        idxs = np.searchsorted(self.ts, ts)

        # Ensure index bounds are valid
        idxs = np.clip(idxs, 1, len(self.ts) - 1)
        left = self.ts[idxs - 1]
        right = self.ts[idxs]

        # Determine whether the left or right value is closer
        idxs -= (np.abs(ts - left) < np.abs(ts - right)).astype(int)

        return self.sampler_class(self._data[idxs])

    def _sample_nearest_before(self, ts):
        last_idx = len(self._data) - 1
        idxs = np.searchsorted(self.ts, ts)
        idxs[idxs > last_idx] = last_idx

        return self.sampler_class(self._data[idxs])

    def _sample_linear_interp(self, sorted_ts):
        result = np.zeros(len(sorted_ts), self.data.dtype)

        for key in self.data.dtype.names:
            result[key] = np.interp(
                sorted_ts, self.ts, self.data[key], left=np.nan, right=np.nan
            )
        return self.sampler_class(result.view(self.data.__class__))

    def __iter__(self):
        for sample in self.data:
            yield sample

    def __len__(self):
        return len(self.ts)

    def to_numpy(self):
        return self._data

    @property
    def data(self):
        return self._data

    @property
    def ts(self):
        return self[TIMESTAMP_FIELD_NAME]


SimpleDataSampler.sampler_class = SimpleDataSampler


class StreamProps:
    ts = proxy[np.int64](TIMESTAMP_FIELD_NAME)
    "The moment these data were recorded"

    def keys(self):
        return dir(self)


class Stream(SimpleDataSampler, StreamProps, Generic[RecordType]):
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
        return self._data.dtype.fields
