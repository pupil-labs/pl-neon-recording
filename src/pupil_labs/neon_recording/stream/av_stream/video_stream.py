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
        idxs = np.searchsorted(self._ts, ts, side="left")
        idxs[idxs > last_idx] = last_idx

        return VideoSampler(self.video_stream, self.ts[idxs])

    def __iter__(self):
        closest_idxs = np.searchsorted(self.video_stream.ts, self.ts, side="left")
        for frame_idx in closest_idxs:
                for video_part in self.video_stream.video_parts:
                    if frame_idx < len(video_part.timestamps):
                        frame = video_part.goto_index(frame_idx)
                        actual_ts = video_part.timestamps[frame_idx]

                        setattr(frame, "ts", actual_ts)
                        yield frame
                        break
                    else:
                        frame_idx -= len(video_part.timestamps)

    @property
    def data(self):
        pass

    @property
    def ts(self):
        return self._ts

    def to_numpy(self):
        return np.array([frame.rgb for frame in self])

class VideoStreamPart:
    def __init__(self, container, timestamps):
        self.container = container
        self.timestamps = timestamps
        self.frame_idx = -1
        self.current_frame = None

    def goto_index(self, frame_idx):
        video = self.container.streams.video[0]

        seek_distance = frame_idx - self.frame_idx - 1
        if seek_distance < 0 or seek_distance > 40:
            target_rel_timestamp = int(frame_idx/video.average_rate)
            self.container.seek(int(target_rel_timestamp*1e6), backward=True)
            self.current_frame = next(self.container.decode(video))
            self.frame_idx = int(self.current_frame.pts * video.time_base * video.average_rate)

        for _ in range(self.frame_idx, frame_idx):
            self.current_frame = next(self.container.decode(video))

        self.frame_idx = frame_idx

        return self.current_frame

class VideoStream(VideoSampler):
    def __init__(self, name, base_name, recording):
        self.name = name
        self._base_name = base_name
        self.recording = recording

        log.info(f"NeonRecording: Loading video: {self._base_name}.")

        self.video_parts = []

        video_files = find_sorted_multipart_files(self.recording._rec_dir, self._base_name, ".mp4")
        self._ts = load_multipart_timestamps([p[1] for p in video_files])
        container_start_idx = 0
        for (video_file, _) in video_files:
            container = plv.open(video_file)
            ts = self._ts[container_start_idx:container_start_idx+container.streams.video[0].frames]
            self.video_parts.append(VideoStreamPart(container, ts))

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
