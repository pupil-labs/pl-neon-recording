import math

import numpy as np

from ... import structlog
from ..stream import Stream
from .av_load import _load_av_container

log = structlog.get_logger(__name__)


class VideoStream(Stream):
    def __init__(self, name, file_name, recording, container=None, ts=None):
        super().__init__(name, recording)
        self._file_name = file_name
        self._backing_container = container
        self._ts = ts
        self._width = None
        self._height = None

        self._load()

    def _load(self):
        # if a backing_container is supplied, then a ts array is usually also supplied
        if self._backing_container is None:
            log.info(f"NeonRecording: Loading video: {self._file_name}.")
            self._backing_container, self._ts = _load_av_container(self._recording._rec_dir, self._file_name)

        self._backing_data = self._backing_container.streams.video[0]
        self._data = self._backing_data.frames
        setattr(self._data, "ts", self._ts)
        self._ts_rel = self._ts - self._recording.start_ts
        setattr(self._data, "ts_rel", self._ts_rel)

        self._width = self._data[0].width
        self._height = self._data[0].height

        # rewind video back to start, to ensure it is synced
        # with audio
        self._backing_container.streams.video[0].seek(0)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def ts_rel(self):
        return self._ts_rel

    def _sample_linear_interp(self, sorted_ts):
        raise NotImplementedError(
            "NeonRecording: Video streams only support nearest neighbor interpolation."
        )

    # from stack overflow:
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    def _sample_nearest(self, sorted_tses):
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
