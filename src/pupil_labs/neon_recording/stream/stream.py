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
        self._data = []
        self.data = []
        self.ts = []
        self.ts_rel = []


    @abc.abstractmethod
    def load(self, rec_dir: pathlib.Path, start_ts: float, file_name: Optional[str] = None) -> None:
        pass


    @abc.abstractmethod
    def to_numpy(self):
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
            raise ValueError("This function can only sample a single timestamp. Use 'sample' for multiple timestamps.")

        if self._ts_oob(ts_wanted):
            return None

        if method == "nearest":
            datum = self.data[np.searchsorted(self.ts, ts_wanted)]
            if np.abs(datum.ts - ts_wanted) < dt:
                return datum
            else:
                return None
        elif method == "nearest_rob":
            diffs = np.abs(self.ts - ts_wanted)

            if np.any(diffs < dt):
                idx = int(np.argmin(diffs))
                return self.data[idx]
            else:
                return None
        elif method == "linear":
            datum = self.interp_data([ts_wanted], "linear")
            if np.abs(datum.ts - ts_wanted) < dt:
                return datum
            else:
                return None


    @abc.abstractmethod
    def interp_data(self, sorted_ts, method="nearest"):
        pass


    def sample(self, tstamps, method="nearest"):
        log.debug("NeonRecording: Sampling timestamps.")

        if len(tstamps) == 1:
            if self._ts_oob(tstamps):
                return None

            if method == "nearest" or method == "linear":
                return self.interp_data([tstamps], method)
            elif method == "nearest_rob":
                return self.sample_rob(tstamps)
        else:
            sorted_ts = np.sort(tstamps)
            return self.interp_data(sorted_ts, method)


    # rob testing linear interpolation by hand
    def sample_rob_interp(self, tstamps):
        log.debug("NeonRecording: Sampling with (rob) linear interpolation.")

        if len(tstamps) == 1:
            if self._ts_oob(tstamps):
                return None

            sorted_ts = [tstamps]
        else:
            # for that perculiar user who does not send in the timestamps in order ;-)
            sorted_ts = np.sort(tstamps)

        ds = self.ts - sorted_ts[:, np.newaxis]
        closest_idxs = np.argmin(np.abs(ds), axis=1)

        interp_data = np.zeros(len(sorted_ts), dtype=[('x', '<f8'), ('y', '<f8'), ('ts', '<f8'), ('ts_rel', '<f8')]).view(np.recarray)
        for ic, ix in enumerate(closest_idxs):
            ts = sorted_ts[ic]
            d = ts - self.ts[ix]

            if np.sign(d) == +1:
                if ix == len(self.ts) - 1:
                    interp_data[ic] = self.data[ix]
                    continue
                else:
                    left_ts = self.ts[ix]
                    right_ts = self.ts[ix+1]

                    left_data = self.data[ix]
                    right_data = self.data[ix+1]
            else:
                if ix == 0:
                    interp_data[ic] = self.data[ix]
                    continue
                else:
                    left_ts = self.ts[ix-1]
                    right_ts = self.ts[ix]

                    left_data = self.data[ix-1]
                    right_data = self.data[ix]

            A = (ts - left_ts)/(right_ts - left_ts)

            if self.name == 'gaze':
                interp_data.x[ic] = left_data.x + A * (right_data.x - left_data.x)
                interp_data.y[ic] = left_data.y + A * (right_data.y - left_data.y)
                interp_data.ts[ic] = left_data.ts + A * (right_data.ts - left_data.ts)
                interp_data.ts_rel[ic] = left_data.ts_rel + A * (right_data.ts_rel - left_data.ts_rel)
            elif self.name == 'imu':
                # just testing gaze is enough
                continue

        return interp_data


    def sample_rob_broadcast(self, tstamps):
        log.debug("NeonRecording: Sampling (rob - broadcast) nearest timestamps.")

        if len(tstamps) == 1:
            if self._ts_oob(tstamps):
                return None

            sorted_tses = [tstamps]
        else:
            # for that perculiar user who does not send in the timestamps in order ;-)
            sorted_tses = np.sort(tstamps)

        ds = self.ts - sorted_tses[:, np.newaxis]
        closest_idxs = np.argmin(np.abs(ds), axis=1)

        return self.data[closest_idxs]


    def sample_rob(self, tstamps):
        log.debug("NeonRecording: Sampling (rob) nearest timestamps.")

        if len(tstamps) == 1:
            if self._ts_oob(tstamps):
                return None

            sorted_tses = [tstamps]
        else:
            # for that perculiar user who does not send in the timestamps in order ;-)
            sorted_tses = np.sort(tstamps)

        closest_idxs = [np.argmin(np.abs(self.ts - curr_ts)) for curr_ts in sorted_tses]
        return self.data[closest_idxs]
