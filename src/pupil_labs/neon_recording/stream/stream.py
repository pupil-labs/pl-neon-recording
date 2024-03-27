import abc
import math
from enum import Enum
from typing import Optional

import numpy as np

from .. import structlog

log = structlog.get_logger(__name__)


class InterpolationMethod(Enum):
    NEAREST = "nearest"
    LINEAR = "linear"


# this implements mpk's idea for a stream of data
# from a neon sensor that can be sampled via ts values
# see here:
# https://www.notion.so/pupillabs/Neon-Recording-Python-Lib-5b247c33e1c74f638af2964fa78018ff?pvs=4
class Stream(abc.ABC):
    def __init__(self, name, recording):
        self.name = name
        self._recording = recording
        self._backing_data = None
        self._data = []
        self._ts = []
        self._ts_rel = None

    @property
    def recording(self):
        return self._recording

    @property
    def data(self):
        return self._data

    @property
    def ts(self):
        return self._ts

    @property
    def ts_rel(self):
        if self._ts_rel is None:
            self._ts_rel = self._ts - self.recording.start_ts

        return self._ts_rel

    @abc.abstractmethod
    def _load(self):
        pass

    @abc.abstractmethod
    def _sample_linear_interp(self, sorted_ts):
        pass

    def __getitem__(self, idxs):
        return self._data[idxs]

    def _ts_oob(self, ts: float):
        return ts < self._ts[0] or ts > self._ts[-1]

    def sample_one(
        self, ts_wanted: float, dt: float = 0.01, method=InterpolationMethod.NEAREST
    ):
        log.debug("NeonRecording: Sampling one timestamp.")

        if self._ts_oob(ts_wanted):
            return None

        if method == InterpolationMethod.NEAREST:
            diffs = np.abs(self._ts - ts_wanted)

            if np.any(diffs < dt):
                idx = int(np.argmin(diffs))
                return self._data[idx]
            else:
                return None
        elif method == InterpolationMethod.LINEAR:
            datum = self._sample_linear_interp([ts_wanted])
            if np.abs(datum.ts - ts_wanted) < dt:
                return datum
            else:
                return None

    def sample(self, tstamps, method=InterpolationMethod.NEAREST):
        log.debug("NeonRecording: Sampling timestamps.")

        # in case they pass one float
        # see note at https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html
        if np.ndim(tstamps) == 0:
            tstamps = [tstamps]

        if len(tstamps) == 1:
            if self._ts_oob(tstamps[0]):
                return None

        sorted_tses = np.sort(tstamps)

        if method == InterpolationMethod.NEAREST:
            return self._sample_nearest(sorted_tses)
        elif method == InterpolationMethod.LINEAR:
            return self._sample_linear_interp(sorted_tses)
        else:
            return ValueError(
                "Only LINEAR and NEAREST methods are supported."
            )

    def _sample_nearest_rob(self, sorted_tses):
        log.debug("NeonRecording: Sampling timestamps with nearest neighbor method.")

        closest_idxs = [
            np.argmin(np.abs(self._ts - curr_ts)) if not self._ts_oob(curr_ts) else None
            for curr_ts in sorted_tses
        ]

        for idx in closest_idxs:
            if idx is not None and not np.isnan(idx):
                yield self._data[int(idx)]
            else:
                yield None

    # from stack overflow:
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    def _sample_nearest(self, sorted_tses):
        # uggcf://jjj.lbhghor.pbz/jngpu?i=FRKKRF5i59b
        log.debug("NeonRecording: Sampling timestamps with nearest neighbor method.")

        closest_idxs = np.searchsorted(self._ts, sorted_tses, side="right")
        for i, ts in enumerate(sorted_tses):
            if self._ts_oob(ts):
                yield None
            else:
                idx = closest_idxs[i]
                if idx > 0 and (
                    idx == len(self._ts)
                    or math.fabs(ts - self._ts[idx - 1]) < math.fabs(ts - self._ts[idx])
                ):
                    yield self._data[int(idx - 1)]
                else:
                    yield self._data[int(idx)]


def sampled_to_numpy(sample_generator):
    fst = next(sample_generator)

    if isinstance(fst, np.record):
        # gaze or imu stream
        samples_np = np.fromiter(sample_generator, dtype=fst.dtype).view(np.recarray)
        return np.hstack([fst, samples_np])
    else:
        # video stream
        frames = [fst.rgb] + [frame.rgb for frame in sample_generator]
        return np.array(frames)
