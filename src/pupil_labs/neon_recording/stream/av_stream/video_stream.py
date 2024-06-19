import numpy as np

from ... import structlog
from ..stream import InterpolationMethod
from ...utils import find_sorted_multipart_files, load_multipart_timestamps

import av

log = structlog.get_logger(__name__)


class TimestampedFrame:
    def __init__(self, frame, ts):
        self._frame = frame
        self.ts = ts

    def __getattr__(self, name):
        return getattr(self._frame, name)

    @property
    def bgr(self):
        return self.to_ndarray(format='bgr24')

    @property
    def gray(self):
        return self.to_ndarray(format='gray')


class VideoSampler:
    def __init__(self, video_stream, sample_timestamps):
        self.video_stream = video_stream
        self._ts = sample_timestamps

        self.frame_generators = {
            vp: vp.container.decode(vp.container.streams.video[0])
            for vp in self.video_stream.video_parts
        }

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
                    frame = video_part.goto_index(frame_idx, self.frame_generators[video_part])
                    actual_ts = video_part.timestamps[frame_idx]

                    ts_frame = TimestampedFrame(frame, actual_ts)
                    yield ts_frame
                    break
                else:
                    frame_idx -= len(video_part.timestamps)

    @property
    def data(self):
        return self.video_stream

    @property
    def ts(self):
        return self._ts

    def to_numpy(self):
        return np.array([frame.rgb for frame in self])


class VideoStreamPart:
    def __init__(self, container, timestamps):
        self.container = container
        self.timestamps = timestamps
        self.current_frame = None

    def goto_index(self, frame_idx, frame_generator):
        video = self.container.streams.video[0]

        seek_distance = frame_idx - self.frame_idx
        if seek_distance < 0 or seek_distance > 40:
            target_rel_timestamp = int(frame_idx / video.average_rate)
            self.container.seek(int(target_rel_timestamp * 1e6), backward=True)
            self.current_frame = next(frame_generator)

        for _ in range(self.frame_idx, frame_idx):
            self.current_frame = next(frame_generator)

        return self.current_frame

    @property
    def frame_idx(self):
        if self.current_frame is None:
            return -1

        video = self.container.streams.video[0]
        return int(self.current_frame.pts * video.time_base * video.average_rate)


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
            container = av.open(video_file)
            ts = self._ts[container_start_idx: container_start_idx + container.streams.video[0].frames]
            self.video_parts.append(VideoStreamPart(container, ts))

            container_start_idx += container.streams.video[0].frames

        self.width = container.streams.video[0].width
        self.height = container.streams.video[0].height

        super().__init__(self, self._ts)


class EyeVideoStream(VideoStream):
    def __init__(self, recording):
        super().__init__("eye", "Neon Sensor Module v1", recording)


class SceneVideoStream(VideoStream):
    def __init__(self, recording):
        super().__init__("scene", "Neon Scene Camera v1", recording)


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
