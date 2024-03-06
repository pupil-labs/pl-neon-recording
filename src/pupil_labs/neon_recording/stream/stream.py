import numpy as np

from .. import structlog
log = structlog.get_logger(__name__)

# this implements mpk's idea for a stream of data
# from a neon sensor that can be sampled via ts values
# see here:
# https://www.notion.so/pupillabs/Neon-Recording-Python-Lib-5b247c33e1c74f638af2964fa78018ff?pvs=4
class Stream():
    def __init__(self, name):
        self.name = name
        self._data = []
        self.data = []
        self.ts = []
        self.ts_rel = []
        self.closest_idxs = []


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

            # if we do not do this, then it is confusing when you later
            # do other work with the stream,
            # but double check with mpk what he meant/wanted
            self.closest_idxs = [idx]

            return self.data[idx]
        else:
            return None


    def sample(self, tstamps) -> 'Stream':
        # Use np.searchsorted to find the insertion points for all bounds at once
        start_indices = np.searchsorted(self.ts, tstamps, side='left')
        end_indices = np.searchsorted(self.ts, tstamps, side='right')[1:]  # Shift to align with start_indices for slicing

        # Initialize a list to hold the slice indices
        self.closest_idxs = []
        for start, end in zip(start_indices, np.append(end_indices, len(self.ts))):
            # Generate the range of indices for this slice
            if start != end:
                indices = np.arange(start, end).tolist()
                self.closest_idxs.extend(indices)
            else:
                self.closest_idxs.append(None)

        # the last one is not needed because we only want the bounded matches
        self.closest_idxs = self.closest_idxs[:-1]

        return self


    def sample_rob(self, tstamps) -> 'Stream':
        log.debug("NeonRecording: Sampling timestamp windows.")

        if len(tstamps) == 1:
            if tstamps < self.ts[0]:
                self.closest_idxs = [None]
            
            if tstamps > self.ts[-1]:
                self.closest_idxs = [None]
            
            self.closest_idxs = np.searchsorted(self.ts, tstamps)
            return self

        # for that perculiar user who does not send in the timestamps in order ;-)
        sorted_tses = np.sort(tstamps)

        self.closest_idxs = []
        for tc in range(1, len(sorted_tses)):
            prior_ts = sorted_tses[tc - 1]
            curr_ts = sorted_tses[tc]

            bounded_tstamps = self.ts[(self.ts >= prior_ts) & (self.ts <= curr_ts)]

            if len(bounded_tstamps) == 0:
                self.closest_idxs.append(None)
                continue

            # just always take the one that is closest to the current timestamp
            closest_ts = bounded_tstamps[-1]

            # this has the feeling of suboptimal
            self.closest_idxs.append(int(np.where(self.ts == closest_ts)[0][0]))

        # the return value does not need to be used by the client, but
        # it is necessary for 'zip'-ping together subsampled Streams
        return self
    

    def __iter__(self) -> 'Stream':
        self.idx = 0
        return self
    

    def __next__(self):
        if self.idx < len(self.closest_idxs):
            idx = self.closest_idxs[self.idx]

            self.idx += 1
            if idx is None:
                return None
            else:
                return self.data[idx]
        else:
            raise StopIteration
