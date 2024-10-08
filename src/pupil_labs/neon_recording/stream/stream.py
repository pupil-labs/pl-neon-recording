from enum import Enum

import numpy as np

from .. import structlog

log = structlog.get_logger(__name__)


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
    sampler_class = None

    def __init__(self, data):
        self._data = data

    def sample(self, tstamps=None, method=InterpolationMethod.NEAREST):
        if tstamps is None:
            tstamps = self.ts

        if np.ndim(tstamps) == 0:
            tstamps = [tstamps]

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
            result[key] = np.interp(sorted_ts, self.ts, self.data[key], left=np.nan, right=np.nan)

        return self.sampler_class(result)

    def __iter__(self):
        for sample in self.data:
            yield sample

    def __len__(self):
        return len(self._data.ts)

    def to_numpy(self):
        return self._data

    @property
    def data(self):
        return self._data

    @property
    def ts(self):
        return self._data['ts']


SimpleDataSampler.sampler_class = SimpleDataSampler


class Stream(SimpleDataSampler):
    def __init__(self, name, recording, data):
        super().__init__(data)
        self.name = name
        self.recording = recording
