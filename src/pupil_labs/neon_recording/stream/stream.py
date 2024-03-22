import abc
import pathlib
from typing import Optional

import numpy as np

from .. import structlog

log = structlog.get_logger(__name__)


# this implements mpk's idea for a stream of data
# from a neon sensor that can be sampled via ts values
# see here:
# https://www.notion.so/pupillabs/Neon-Recording-Python-Lib-5b247c33e1c74f638af2964fa78018ff?pvs=4
class Stream(abc.ABC):
    def __init__(self, name):
        self.name = name
        self._backing_data = []
        self.data = []
        self.ts = []
        self.ts_rel = []

    @abc.abstractmethod
    def load(
        self, rec_dir: pathlib.Path, start_ts: float, file_name: Optional[str] = None
    ) -> None:
        pass

    @abc.abstractmethod
    def linear_interp(self, sorted_ts):
        pass

    def __getitem__(self, idxs):
        return self.data[idxs]

    def _ts_oob(self, ts: float):
        return ts < self.ts[0] or ts > self.ts[-1]

    def sample_one(self, ts_wanted: float, dt: float = 0.01, method="nearest"):
        log.debug("NeonRecording: Sampling one timestamp.")

        # in case they pass multiple timestamps
        # see note at https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html
        if not np.ndim(ts_wanted) == 0:
            raise ValueError(
                "This function can only sample a single timestamp. Use 'sample' for multiple timestamps."
            )

        if self._ts_oob(ts_wanted):
            return None

        if method == "insert_order":
            datum = self.data[np.searchsorted(self.ts, ts_wanted)]
            if np.abs(datum.ts - ts_wanted) < dt:
                return datum
            else:
                return None
        elif method == "nearest":
            diffs = np.abs(self.ts - ts_wanted)

            if np.any(diffs < dt):
                idx = int(np.argmin(diffs))
                return self.data[idx]
            else:
                return None
        elif method == "linear":
            datum = self.linear_interp([ts_wanted])
            if np.abs(datum.ts - ts_wanted) < dt:
                return datum
            else:
                return None

    def sample(self, tstamps, method="nearest"):
        log.debug("NeonRecording: Sampling timestamps.")

        # in case they pass one float
        # see note at https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html
        if np.ndim(tstamps) == 0:
            tstamps = [tstamps]

        if len(tstamps) == 1:
            if self._ts_oob(tstamps[0]):
                return None

        sorted_ts = np.sort(tstamps)

        if method == "linear":
            return self.linear_interp(sorted_ts)
        elif method == "nearest":
            return self.sample_nearest(sorted_ts)
        elif method == "insert_order":
            return self.sample_sorted(sorted_ts)
        else:
            return ValueError(
                "Only 'linear', 'nearest', and 'insert_order' methods are supported."
            )

    # this works for all the different streams, so define it here, rather than in multiple different places
    def sample_nearest(self, sorted_tses):
        log.debug("NeonRecording: Sampling nearest timestamps.")

        closest_idxs = [
            np.argmin(np.abs(self.ts - curr_ts)) if not self._ts_oob(curr_ts) else None
            for curr_ts in sorted_tses
        ]

        for idx in closest_idxs:
            if idx is not None and not np.isnan(idx):
                yield self.data[int(idx)]
            else:
                yield None

    def sample_sorted(self, sorted_tses):
        # uggcf://jjj.lbhghor.pbz/jngpu?i=FRKKRF5i59b
        log.debug("NeonRecording: Sampling sorted timestamps.")

        closest_idxs = np.searchsorted(self.ts, sorted_tses)
        for i, ts in enumerate(sorted_tses):
            if self._ts_oob(ts):
                closest_idxs[i] = np.nan

        for idx in closest_idxs:
            if idx is not None and not np.isnan(idx):
                yield self.data[int(idx)]
            else:
                yield None


def subsampled_to_numpy(sample_generator):
    fst = next(sample_generator)

    if isinstance(fst, np.record):
        # gaze or imu stream
        samples_np = np.fromiter(sample_generator, dtype=fst.dtype).view(np.recarray)
        return np.hstack([fst, samples_np])
    else:
        # video stream
        frames = [fst.rgb] + [frame.rgb for frame in sample_generator]
        return np.array(frames)
