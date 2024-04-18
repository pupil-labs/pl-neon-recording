import abc
import math
from enum import Enum

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
        self.recording = recording
        self.data = None

    @property
    def ts(self):
        return self.data.ts

    @abc.abstractmethod
    def _sample_linear_interp(self, sorted_ts):
        pass

    def __getitem__(self, idxs):
        if isinstance(idxs, tuple):
            if isinstance(idxs[0], slice):
                start = idxs[0].start or 0
                stop = idxs[0].stop or len(self.data)
                step = idxs[0].step or 1

                rows_idxs = list(range(start, stop, step))

            if isinstance(idxs[1], slice):
                names = self.data.dtype.names
                start = idxs[0].start or 0
                stop = idxs[0].stop or len(names)
                step = idxs[0].step or 1
                column_names = names[start:stop:step]

            else:
                column_names = [self.data.dtype.names[idxs[1]]]

        else:
            rows_idxs = idxs
            column_names = self.data.dtype.names

        rows = self.data[rows_idxs]
        return rows[list(column_names)]


    def ts_oob(self, ts: float):
        return ts < self.ts[0] or ts > self.ts[-1]

    def sample_one(
        self, ts_wanted: float, dt: float = 0.01, method=InterpolationMethod.NEAREST
    ):
        if self.ts_oob(ts_wanted):
            return None

        if method == InterpolationMethod.NEAREST:
            diffs = np.abs(self.ts - ts_wanted)

            if np.any(diffs < dt):
                idx = int(np.argmin(diffs))
                return self.data[idx]
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

        if np.ndim(tstamps) == 0:
            tstamps = [tstamps]

        if len(tstamps) == 1:
            if self.ts_oob(tstamps[0]):
                return None

        sorted_tses = np.sort(tstamps)

        if method == InterpolationMethod.NEAREST:
            return self._sample_nearest(sorted_tses)

        elif method == InterpolationMethod.LINEAR:
            return self._sample_linear_interp(sorted_tses)

    # from stack overflow:
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    def _sample_nearest(self, sorted_tses):
        log.debug("NeonRecording: Sampling timestamps with nearest neighbor method.")

        closest_idxs = np.searchsorted(self.ts, sorted_tses, side="right")
        for i, ts in enumerate(sorted_tses):
            if self.ts_oob(ts):
                yield None

            else:
                idx = closest_idxs[i]
                if idx > 0 and (
                    idx == len(self.ts)
                    or math.fabs(ts - self.ts[idx - 1]) < math.fabs(ts - self.ts[idx])
                ):
                    yield self.data[int(idx - 1)]
                else:
                    yield self.data[int(idx)]


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
