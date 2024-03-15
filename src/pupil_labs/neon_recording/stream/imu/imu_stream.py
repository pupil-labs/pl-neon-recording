import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from ..stream import Stream
from .load_imu_data import IMURecording

from ... import structlog
log = structlog.get_logger(__name__)

class IMUStream(Stream):
    def linear_interp(self, sorted_ts):
        interp_data = np.zeros(len(sorted_ts), dtype=IMURecording.DTYPE_RAW).view(np.recarray)
        for field in IMURecording.DTYPE_RAW.names:
            if field == "ts":
                interp_data[field] = sorted_ts

            interp_data[field] = np.interp(sorted_ts, self.ts, self.data[field], left=np.nan, right=np.nan)

        def sample_gen():
            for d in interp_data:
                if not np.isnan(d.gyro_x):
                    yield self.data[idx]
                else:
                    yield None

            return sample_gen()


    def load(self, rec_dir: pathlib.Path, start_ts: float, file_name: Optional[str] = None) -> None:
        imu_rec = IMURecording(rec_dir / 'extimu ps1.raw', start_ts)

        self._data = imu_rec.raw
        self.data = self._data[:]
        self.ts = self._data[:].ts
        self.ts_rel = self._data[:].ts_rel


    def to_numpy(self):
        log.debug("NeonRecording: Converting sampled IMU data to numpy array.")

        cid = self.data[self.closest_idxs]
        return pd.DataFrame(cid).to_numpy()
