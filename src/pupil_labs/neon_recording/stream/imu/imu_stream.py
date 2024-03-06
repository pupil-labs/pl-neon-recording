import pathlib

import numpy as np
import pandas as pd

from ..stream import Stream
from .load_imu_data import IMURecording

from ... import structlog
log = structlog.get_logger(__name__)

class IMUStream(Stream):
    def to_numpy(self):
        log.debug("NeonRecording: Converting sampled IMU data to numpy array.")

        cid = self.data[self.closest_idxs]
        return pd.DataFrame(cid).to_numpy()
    

    def load(self, rec_dir: pathlib.Path, start_ts: float) -> None:
        imu_rec = IMURecording(rec_dir / 'extimu ps1.raw', start_ts)

        self._data = imu_rec.raw
        self.data = self._data[:]
        self.ts = self._data[:].ts
        self.ts_rel = self._data[:].ts_rel

        # because if they use to_numpy without calling sample,
        # then they still get a sensible result: all the data
        self.closest_idxs = np.arange(len(self.data))
