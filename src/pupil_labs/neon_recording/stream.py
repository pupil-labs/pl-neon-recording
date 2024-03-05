import numpy as np
import pandas as pd

# this implements mpk's idea for a stream of data
# from a neon sensor that can be sampled via ts values
# see here:
# https://www.notion.so/pupillabs/Neon-Recording-Python-Lib-5b247c33e1c74f638af2964fa78018ff?pvs=4
class Stream():
    def __init__(self, name):
        self.name = name
        self.data = []
        self.ts = []
        self.ts_rel = []
        self.closest_idxs = []


    def load(self, data):
        self.data = data[:]
        self.ts = data[:].ts
        self.ts_rel = data[:].ts_rel

        # because if they use to_numpy without calling sample,
        # then they still get a sensible result: all the data
        self.closest_idxs = np.arange(len(self.data))

    
    def to_numpy(self):
        cid = self.data[self.closest_idxs]
        return pd.DataFrame(cid).to_numpy()
    

    def __getitem__(self, idxs):
        return self.data[idxs]


    def sample_one(self, ts_wanted, dt = 0.01):
        # in case they pass multiple timestamps
        # see note at https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html
        if not np.ndim(ts_wanted) == 0:
            raise ValueError("This function can only sample a single timestamp. Use 'sample' for multiple timestamps.")

        diffs = np.abs(self.ts - ts_wanted)

        if np.any(diffs < dt):
            idx = np.argmin(diffs)

            # if we do not do this, then it is confusing when you later
            # do other work with the stream,
            # but double check with mpk what he meant/wanted
            self.closest_idxs = [idx]

            return self.data[idx]
        else:
            return None


    def sample(self, tstamps):
        if len(tstamps) == 1:
            if tstamps < self.ts[0]:
                self.closest_idxs = [None]
            
            if tstamps > self.ts[-1]:
                self.closest_idxs = [None]
            
            self.closest_idxs = np.searchsorted(self.ts, tstamps)

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
            self.closest_idxs.append(np.where(self.ts == closest_ts)[0][0])

        # the return value does not need to be used by the client, but
        # it is necessary for 'zip'-ping together subsampled Streams
        return self
    

    def __iter__(self):
        self.idx = 0
        return self

    
    def __next__(self):
        if self.idx < len(self.closest_idxs):
            idx = self.closest_idxs[self.idx]

            if idx is None:
                self.idx += 1
                return None
            else:
                self.idx += 1
                return self.data[idx:idx+1]
        else:
            raise StopIteration