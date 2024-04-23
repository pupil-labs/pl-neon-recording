import math

import numpy as np

from ... import structlog
from ..stream import InterpolationMethod
from ...utils import find_sorted_multipart_files, load_multipart_timestamps

import pupil_labs.video as plv

log = structlog.get_logger(__name__)


class VideoSampler:
    def __init__(self, video_stream, sample_timestamps):
        self.video_stream = video_stream
        self._ts = sample_timestamps


    def sample(self, tstamps, method=InterpolationMethod.NEAREST):
        return self._sample_nearest(tstamps)

    def _sample_nearest(self, ts):
        last_idx = len(self._ts) - 1
        idxs = np.searchsorted(self._ts, ts, side="right")
        idxs[idxs > last_idx] = last_idx

        return VideoSampler(self.video_stream, self.ts[idxs])

    def __iter__(self):
        closest_idxs = np.searchsorted(self.video_stream.ts, self.ts, side="right")
        for frame_idx, requested_ts in zip(closest_idxs, self.ts):
                for (container, timestamps) in self.video_stream._video_ts_pairs:
                    if requested_ts < timestamps[-1]:
                        actual_ts = timestamps[frame_idx]

                        video = container.streams.video[0]

                        target_rel_timestamp = int(frame_idx/video.average_rate)
                        container.seek(int(target_rel_timestamp*1e6), backward=True)

                        frame = next(container.decode(video))
                        current_frame_idx = int(frame.pts * video.time_base * video.average_rate)
                        for _ in range(current_frame_idx, frame_idx):
                            frame = next(container.decode(video))

                        setattr(frame, "ts", actual_ts)
                        yield frame
                        break
                    else:
                        frame_idx -= len(timestamps)

    @property
    def data(self):
        pass

    @property
    def ts(self):
        return self._ts

    def to_numpy(self):
        return np.array([frame.rgb for frame in self])



class VideoStream(VideoSampler):
    def __init__(self, name, base_name, recording):
        self.name = name
        self._base_name = base_name
        self.recording = recording

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

        super().__init__(self, self._ts)


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
