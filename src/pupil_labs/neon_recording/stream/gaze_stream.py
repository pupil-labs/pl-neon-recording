import pathlib
from typing import Optional

import numpy as np

from .. import structlog
from ..time_utils import load_and_convert_tstamps
from .stream import Stream

log = structlog.get_logger(__name__)


def _convert_gaze_data_to_recarray(gaze_data, ts, ts_rel):
    log.debug("NeonRecording: Converting gaze data to recarray format.")

    if gaze_data.shape[0] != len(ts):
        log.error("NeonRecording: Length mismatch - gaze_data and ts.")
        raise ValueError("gaze_data and ts must have the same length")
    if len(ts) != len(ts_rel):
        log.error("NeonRecording: Length mismatch - ts and ts_rel.")
        raise ValueError("ts and ts_rel must have the same length")

    out = np.recarray(
        gaze_data.shape[0],
        dtype=[("x", "<f8"), ("y", "<f8"), ("ts", "<f8"), ("ts_rel", "<f8")],
    )
    out.x = gaze_data[:, 0]
    out.y = gaze_data[:, 1]
    out.ts = ts.astype(np.float64)
    out.ts_rel = ts_rel.astype(np.float64)

    return out


class GazeStream(Stream):
    def __init__(self, name, recording):
        super().__init__(name, recording)
        self._load()

    def _sample_linear_interp(self, sorted_ts):
        xs = self._data.x
        ys = self._data.y

        interp_data = np.zeros(
            len(sorted_ts),
            dtype=[("x", "<f8"), ("y", "<f8"), ("ts", "<f8"), ("ts_rel", "<f8")],
        ).view(np.recarray)
        interp_data.x = np.interp(sorted_ts, self._ts, xs, left=np.nan, right=np.nan)
        interp_data.y = np.interp(sorted_ts, self._ts, ys, left=np.nan, right=np.nan)
        interp_data.ts = sorted_ts
        interp_data.ts_rel = np.interp(
            sorted_ts, self._ts, self._ts_rel, left=np.nan, right=np.nan
        )

        for d in interp_data:
            if not np.isnan(d.x):
                yield d
            else:
                yield None

    def _load(self) -> None:
        log.info("NeonRecording: Loading gaze data")

        # we use gaze_200hz from cloud for the rec gaze stream
        # ts, raw = self._load_ts_and_data(rec_dir, 'gaze ps1')
        gaze_200hz_ts, gaze_200hz_raw = self._load_ts_and_data("gaze_200hz")
        gaze_200hz_ts_rel = gaze_200hz_ts - self._recording._start_ts

        # load up raw timestamps in original ns format,
        # in case useful at some point
        log.debug("NeonRecording: Loading raw gaze timestamps (ns)")
        self._gaze_ps1_raw_time_ns = np.fromfile(
            str(self._recording._rec_dir / "gaze ps1.time"), dtype="<u8"
        )
        self._gaze_200hz_raw_time_ns = np.fromfile(
            str(self._recording._rec_dir / "gaze_200hz.time"), dtype="<u8"
        )

        # still not sure what gaze_right is...
        # log.info("NeonRecording: Loading 'gaze_right_ps1' data")
        # rec._gaze_right_ps1_ts, rec._gaze_right_ps1_raw = _load_ts_and_data(self._recording._rec_dir, 'gaze ps1')

        data = _convert_gaze_data_to_recarray(
            gaze_200hz_raw, gaze_200hz_ts, gaze_200hz_ts_rel
        )

        self._data = data
        self._ts = self._data[:].ts

    def _load_ts_and_data(self, stream_filename: str):
        log.debug("NeonRecording: Loading gaze data and timestamps.")

        try:
            ts = load_and_convert_tstamps(
                self._recording._rec_dir / (stream_filename + ".time")
            )
        except Exception as e:
            log.exception(f"Error loading timestamps: {e}")
            raise

        try:
            raw = self._load_gaze_raw_data(
                self._recording._rec_dir / (stream_filename + ".raw")
            )
        except Exception as e:
            log.exception(f"Error loading raw data: {e}")
            raise

        return ts, raw

    def _load_gaze_raw_data(self, path: pathlib.Path):
        log.debug("NeonRecording: Loading gaze raw data.")

        raw_data = np.fromfile(str(path), "<f4")
        raw_data = raw_data.reshape((-1, 2))
        return np.asarray(raw_data, dtype=raw_data.dtype).astype(np.float64)
