import math

import numpy as np

from ... import structlog
from ..stream import Stream
from ...utils import find_sorted_multipart_files, load_multipart_timestamps

import pupil_labs.video as plv

log = structlog.get_logger(__name__)

class VideoStream(Stream):
    def __init__(self, name, base_name, recording):
        super().__init__(name, recording)
        self._base_name = base_name
        log.info(f"NeonRecording: Loading video: {self._base_name}.")

        self._video_ts_pairs = []

        video_files = find_sorted_multipart_files(self.recording._rec_dir, self._base_name, ".mp4")
        self._ts = load_multipart_timestamps([p[1] for p in video_files])
        container_start_idx = 0
        for (video_file, _) in video_files:
            container = plv.open(video_file)
            ts = self._ts[container_start_idx:container_start_idx+container.streams.video[0].frames]
            self._video_ts_pairs.append((container, ts))

            container_start_idx += container.streams.video[0].frames

        self.width = container.streams.video[0].width
        self.height = container.streams.video[0].height

    def _sample_linear_interp(self, sorted_ts):
        raise NotImplementedError(
            "NeonRecording: Video streams only support nearest neighbor interpolation."
        )

    def _sample_nearest(self, sorted_tses, epsilon):
        log.debug("NeonRecording: Sampling nearest timestamps.")

        closest_idxs = np.searchsorted(self._ts, sorted_tses, side="right")
        for frame_idx, requested_ts in zip(closest_idxs, sorted_tses):
            if self.ts_oob(requested_ts):
                yield None

            else:
                frame_idx = int(frame_idx)
                for (video, timestamps) in self._video_ts_pairs:

                    if requested_ts < timestamps[-1]:
                        actual_ts = timestamps[frame_idx]

                        if epsilon is not None and abs(actual_ts - requested_ts) > epsilon:
                            yield GrayFrame(self.width, self.height)
                            break

                        video.seek(frame_idx)
                        frame = next(video.decode(video.streams.video[0]))
                        setattr(frame, "ts", actual_ts)
                        yield frame
                        break
                    else:
                        frame_idx -= len(timestamps)

    @property
    def ts(self):
        return self._ts

class GrayFrame():
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self._bgr = None
        self._gray = None

    @property
    def bgr(self):
        if self._bgr is None:
            self._bgr = 128 * np.ones([self.height, self.width, 3], dtype='uint8')

        return self._bgr

    @property
    def gray(self):
        if self._gray is None:
            self._gray = 128 * np.ones([self.height, self.width], dtype='uint8')

        return self._gray
