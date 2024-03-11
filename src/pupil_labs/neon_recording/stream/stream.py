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
    

    def sample_one(self, ts_wanted: float, dt: float = 0.01):
        log.debug("NeonRecording: Sampling one timestamp.")

        # in case they pass multiple timestamps
        # see note at https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html
        if not np.ndim(ts_wanted) == 0:
            raise ValueError("This function can only sample a single timestamp. Use 'sample' for multiple timestamps.")

        diffs = np.abs(self.ts - ts_wanted)

        if np.any(diffs < dt):
            idx = int(np.argmin(diffs))
            return self.data[idx]
        else:
            return None
        

    @abc.abstractmethod
    def interp_data(self, sorted_ts, method="nearest"):
        pass


    def sample(self, tstamps, method="nearest"):
        log.debug("NeonRecording: Sampling timestamps.")

        if len(tstamps) == 1:
            if method == "nearest":
                return self.data[np.searchsorted(self.ts, tstamps)]
            elif method == "linear":
                return self.interp_data([tstamps], "linear")
        else:
            sorted_ts = np.sort(tstamps)
            return self.interp_data(sorted_ts, method)


    def sample_rob_interp(self, tstamps):
        log.debug("NeonRecording: Sampling with (rob) linear interpolation.")

        # rob testing linear interpolation by hand
        sorted_ts = np.sort(tstamps)
        
        ds = self.ts - sorted_ts[:, np.newaxis]
        closest_idxs = np.argmin(np.abs(ds), axis=1)

        interp_data = np.zeros(len(sorted_ts), dtype=[('x', '<f8'), ('y', '<f8'), ('ts', '<f8'), ('ts_rel', '<f8')]).view(np.recarray)
        for ic, ix in enumerate(closest_idxs):
            ts = sorted_ts[ic]
            d = ts - self.ts[ix]

            if np.sign(d) == +1:
                left_ts = self.ts[ix]
                right_ts = self.ts[ix+1]

                left_data = self.data[ix]
                right_data = self.data[ix+1]
            else:
                left_ts = self.ts[ix-1]
                right_ts = self.ts[ix]

                left_data = self.data[ix-1]
                right_data = self.data[ix]

            A = (ts - left_ts)/(right_ts - left_ts)

            interp_data.x[ic] = left_data.x + A * (right_data.x - left_data.x)
            interp_data.y[ic] = left_data.y + A * (right_data.y - left_data.y)
            interp_data.ts[ic] = left_data.ts + A * (right_data.ts - left_data.ts)
            interp_data.ts_rel[ic] = left_data.ts_rel + A * (right_data.ts_rel - left_data.ts_rel)

        return interp_data


    def sample_mpk(self, tstamps):
        log.debug("NeonRecording: Sampling (mpk) timestamp windows.")

        # Use np.searchsorted to find the insertion points for all bounds at once
        start_indices = np.searchsorted(self.ts, tstamps, side='left')
        # Shift to align with start_indices for slicing
        end_indices = np.searchsorted(self.ts, tstamps, side='right')[1:]

        # Initialize a list to hold the slice indices
        closest_idxs = []
        for start, end in zip(start_indices, np.append(end_indices, len(self.ts))):
            # Generate the range of indices for this slice
            if start != end:
                indices = np.arange(start, end)
                closest_idxs.append(indices)

        # the last one is not needed because we only want the bounded matches
        closest_idxs = closest_idxs[:-1]
        closest_idxs = np.array([idx.flatten() for idx in closest_idxs if len(idx) > 0])
        closest_idxs = np.unique(closest_idxs)

        return self.data[closest_idxs]


    def sample_rob_broadcast(self, tstamps):
        log.debug("NeonRecording: Sampling (rob - broadcast) nearest timestamps.")

        if len(tstamps) == 1:
            return self.data[np.searchsorted(self.ts, tstamps)]

        # for that perculiar user who does not send in the timestamps in order ;-)
        sorted_tses = np.sort(tstamps)
        ds = self.ts - sorted_tses[:, np.newaxis]
        closest_idxs = np.argmin(np.abs(ds), axis=1)

        return self.data[closest_idxs]


    def sample_rob(self, tstamps):
        log.debug("NeonRecording: Sampling (rob) nearest timestamps.")

        if len(tstamps) == 1:
            return self.data[np.searchsorted(self.ts, tstamps)]

        # for that perculiar user who does not send in the timestamps in order ;-)
        sorted_tses = np.sort(tstamps)
        closest_idxs = [np.argmin(np.abs(self.ts - curr_ts)) for curr_ts in sorted_tses]

        return self.data[closest_idxs]
    

    def sample_rob_orig(self, tstamps):
        log.debug("NeonRecording: Sampling (rob) timestamp windows.")

        if len(tstamps) == 1:
            return self.data[np.searchsorted(self.ts, tstamps)]

        # for that perculiar user who does not send in the timestamps in order ;-)
        sorted_tses = np.sort(tstamps)

        closest_idxs = []
        for tc in range(1, len(sorted_tses)):
            prior_ts = sorted_tses[tc - 1]
            curr_ts = sorted_tses[tc]

            if tc == 1:
                bounded_tstamps = self.ts[(self.ts >= prior_ts) & (self.ts <= curr_ts)]
            else:
                bounded_tstamps = self.ts[(self.ts > prior_ts) & (self.ts <= curr_ts)]

            # just always take the one that is closest to the current timestamp
            closest_ts = bounded_tstamps[-1]

            # this has the feeling of suboptimal
            closest_idxs.append(int(np.where(self.ts == closest_ts)[0][0]))

        return self.data[closest_idxs]
