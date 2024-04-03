import numpy as np

from ... import structlog
from ..stream import Stream
from .load_imu_data import IMURecording

log = structlog.get_logger(__name__)


class IMUStream(Stream):
    def __init__(self, name, recording):
        super().__init__(name, recording)
        self._load()

    def _sample_linear_interp(self, sorted_ts):
        interp_data = np.zeros(len(sorted_ts), dtype=IMURecording.DTYPE_RAW).view(
            np.recarray
        )
        for field in IMURecording.DTYPE_RAW.names:
            if field == "ts":
                interp_data[field] = sorted_ts

            interp_data[field] = np.interp(
                sorted_ts, self._ts, self._data[field], left=np.nan, right=np.nan
            )

        for d in interp_data:
            if not np.isnan(d.gyro_x):
                yield d
            else:
                yield None

    def _load(self):
        log.info("NeonRecording: Loading IMU data")

        imu_rec = IMURecording(
            self._recording._rec_dir / "extimu ps1.raw", self._recording._start_ts
        )

        self._data = imu_rec.raw
        self._ts = self._data[:].ts
