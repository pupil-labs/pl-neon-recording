import numpy as np
import pandas as pd

import pupil_labs.video as plv

# this implements mpk's idea for a stream of data
# from a neon sensor that can be sampled via ts values
# see here:
# https://www.notion.so/pupillabs/Neon-Recording-Python-Lib-5b247c33e1c74f638af2964fa78018ff?pvs=4
class Stream():
    def __init__(self, name):
        self.name = name
        self.data = []
        self._ts = []
        self._ts_rel = []
        self.closest_idxs = []


    # TODO(rob) - find a good way to return sub-sampled tses
    @property
    def ts(self):
        return self._ts

    @ts.setter
    def ts(self, tstamps):
        self._ts = tstamps


    @property
    def ts_rel(self):
        return self._ts_rel

    @ts_rel.setter
    def ts_rel(self, tstamps_rel):
        self._ts_rel = tstamps_rel


    def load(self, data):
        self.data = data[:]
        self._ts = data[:].ts
        self._ts_rel = data[:].ts_rel

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

        diffs = np.abs(self._ts - ts_wanted)

        if np.any(diffs < dt):
            idx = int(np.argmin(diffs))

            # if we do not do this, then it is confusing when you later
            # do other work with the stream,
            # but double check with mpk what he meant/wanted
            self.closest_idxs = [idx]

            return self.data[idx]
        else:
            return None


    def sample(self, tstamps):
        if len(tstamps) == 1:
            if tstamps < self._ts[0]:
                self.closest_idxs = [None]
            
            if tstamps > self._ts[-1]:
                self.closest_idxs = [None]
            
            self.closest_idxs = np.searchsorted(self._ts, tstamps)

        # for that perculiar user who does not send in the timestamps in order ;-)
        sorted_tses = np.sort(tstamps)

        self.closest_idxs = []
        for tc in range(1, len(sorted_tses)):
            prior_ts = sorted_tses[tc - 1]
            curr_ts = sorted_tses[tc]

            bounded_tstamps = self._ts[(self._ts >= prior_ts) & (self._ts <= curr_ts)]

            if len(bounded_tstamps) == 0:
                self.closest_idxs.append(None)
                continue

            # just always take the one that is closest to the current timestamp
            closest_ts = bounded_tstamps[-1]

            # this has the feeling of suboptimal
            self.closest_idxs.append(int(np.where(self._ts == closest_ts)[0][0]))

        # the return value does not need to be used by the client, but
        # it is necessary for 'zip'-ping together subsampled Streams
        return self
    

    def __iter__(self):
        self.idx = 0
        return self

    
    def __next__(self):
        if self.idx < len(self.closest_idxs):
            idx = self.closest_idxs[self.idx]

            self.idx += 1
            if idx is None:
                return None
            else:
                return self.data[idx:idx+1]
        else:
            raise StopIteration
        

class VideoStream(Stream):
    # here data will be a dict with video container and tstamps
    def load(self, data):
        self.data = data['av_container'].streams.video[0]
        self._ts = data['ts']
        self._ts_rel = data['ts_rel']

        # because if they use to_numpy without calling sample,
        # then they still get a sensible result: all the data
        self.closest_idxs = np.arange(len(self._ts))


    # TODO(rob) - probably need a different way here
    # def to_numpy(self):
    #     cid = self.data[self.closest_idxs]
    #     return pd.DataFrame(cid).to_numpy()


    def sample_one(self, ts_wanted, dt = 0.01):
        # in case they pass multiple timestamps
        # see note at https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html
        if not np.ndim(ts_wanted) == 0:
            raise ValueError("This function can only sample a single timestamp. Use 'sample' for multiple timestamps.")

        diffs = np.abs(self._ts - ts_wanted)

        if np.any(diffs < dt):
            idx = int(np.argmin(diffs))

            # if we do not do this, then it is confusing when you later
            # do other work with the stream,
            # but double check with mpk what he meant/wanted
            self.closest_idxs = [idx]

            # minor change to account for video stream format
            return self.data.frames[idx]
        else:
            return None


    def __getitem__(self, idxs):
        return self.data.frames[idxs]

    
    def __next__(self):
        if self.idx < len(self.closest_idxs):
            idx = self.closest_idxs[self.idx]

            self.idx += 1
            if idx is None:
                return None
            else:
                # minor change to account for video stream format
                return self.data.frames[idx]
        else:
            raise StopIteration
