import pathlib
from typing import Optional

import numpy as np

from ... import structlog
from ..stream import Stream
from .load_imu_data import IMURecording

log = structlog.get_logger(__name__)


class IMUStream(Stream):
    def linear_interp(self, sorted_ts):
        interp_data = np.zeros(len(sorted_ts), dtype=IMURecording.DTYPE_RAW).view(
            np.recarray
        )
        for field in IMURecording.DTYPE_RAW.names:
            if field == "ts":
                interp_data[field] = sorted_ts

            interp_data[field] = np.interp(
                sorted_ts, self.ts, self.data[field], left=np.nan, right=np.nan
            )

        for d in interp_data:
            if not np.isnan(d.gyro_x):
                yield d
            else:
                yield None

    def load(
        self, rec_dir: pathlib.Path, start_ts: float, file_name: Optional[str] = None
    ) -> None:
        imu_rec = IMURecording(rec_dir / "extimu ps1.raw", start_ts)

        self._backing_data = imu_rec.raw
        self.data = self._backing_data[:]
        self.ts = self._backing_data[:].ts
        self.ts_rel = self._backing_data[:].ts_rel
