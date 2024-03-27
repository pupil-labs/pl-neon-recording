import math
from typing import Optional

import numpy as np
import pupil_labs.video as plv

from .. import structlog
from ..time_utils import load_and_convert_tstamps
from .stream import Stream

log = structlog.get_logger(__name__)


class VideoStream(Stream):
    def __init__(self, name, file_name, recording):
        super().__init__(name, recording)
        self._file_name = file_name
        self._width = None
        self._height = None

        self._load()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def ts_rel(self):
        # if self._ts_rel is None:
        # self._ts_rel = self._ts - self._recording._start_ts
        # setattr(self._data, "ts_rel", self._ts_rel)

        return self._ts_rel

    def _sample_linear_interp(self, sorted_ts):
        raise NotImplementedError(
            "NeonRecording: Video streams only support nearest neighbor interpolation."
        )

    def _load(self):
        log.info(f"NeonRecording: Loading video: {self._file_name}.")

        container, ts = self._load_video(self._file_name)

        self._backing_data = container.streams.video[0]
        self._data = self._backing_data.frames
        self._ts = ts
        setattr(self._data, "ts", self._ts)
        self._ts_rel = self._ts - self._recording._start_ts

        self._width = self._data[0].width
        self._height = self._data[0].height

    def _load_video(self, video_name: str):
        log.debug(
            f"NeonRecording: Loading video and associated timestamps: {video_name}."
        )

        if not (self._recording._rec_dir / (video_name + ".mp4")).exists():
            raise FileNotFoundError(
                f"File not found: {self._recording._rec_dir / (video_name + '.mp4')}. Please double check the recording download."
            )

        container = plv.open(self._recording._rec_dir / (video_name + ".mp4"))

        # use hardware ts
        # ts = load_and_convert_tstamps(self._recording._rec_dir / (video_name + '.time_aux'))
        try:
            ts = load_and_convert_tstamps(
                self._recording._rec_dir / (video_name + ".time")
            )
        except Exception as e:
            log.exception(f"Error loading timestamps: {e}")
            raise

        return container, ts

    def _sample_nearest_rob(self, sorted_tses):
        log.debug("NeonRecording: Sampling nearest timestamps.")

        closest_idxs = [
            np.argmin(np.abs(self._ts - curr_ts)) if not self._ts_oob(curr_ts) else None
            for curr_ts in sorted_tses
        ]

        for idx in closest_idxs:
            if idx is not None and not np.isnan(idx):
                d = self._data[int(idx)]
                setattr(d, "ts", self._ts[int(idx)])
                setattr(d, "ts_rel", self._ts_rel[int(idx)])
                yield d
            else:
                yield None

    # from stack overflow:
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    def _sample_nearest(self, sorted_tses):
        # uggcf://jjj.lbhghor.pbz/jngpu?i=FRKKRF5i59b
        log.debug("NeonRecording: Sampling nearest timestamps.")

        closest_idxs = np.searchsorted(self._ts, sorted_tses, side="right")
        for i, ts in enumerate(sorted_tses):
            if self._ts_oob(ts):
                yield None
            else:
                idx = closest_idxs[i]
                if idx > 0 and (
                    idx == len(self._ts)
                    or math.fabs(ts - self._ts[idx - 1]) < math.fabs(ts - self._ts[idx])
                ):
                    idx = idx - 1

                d = self._data[int(idx)]
                setattr(d, "ts", self._ts[int(idx)])
                setattr(d, "ts_rel", self._ts_rel[int(idx)])
                yield d
