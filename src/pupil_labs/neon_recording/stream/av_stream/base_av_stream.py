import numpy as np

from ..stream import InterpolationMethod
from ... import structlog


log = structlog.get_logger(__name__)


class TimestampedFrame:
    """
    Wraps either a :class:`av.video.frame.VideoFrame` or an :class:`av.audio.frame.AudioFrame`
    """

    def __init__(self, frame, ts):
        self._frame = frame
        self._ts = ts

    def __getattr__(self, name):
        return getattr(self._frame, name)

    @property
    def ts(self):
        return self._ts

    @property
    def bgr(self):
        return self.to_ndarray(format='bgr24')

    @property
    def gray(self):
        return self.to_ndarray(format='gray')


class StreamSampler:
    """
    """

    def __init__(self, raw_stream, sample_timestamps, audio_or_video):
        self.raw_stream = raw_stream
        self._ts = sample_timestamps
        self.audio_or_video = audio_or_video

        self.frame_generators = {
            vp: vp.container.decode(getattr(vp.container.streams, audio_or_video)[0])
            for vp in self.raw_stream.video_parts
        }

    def sample(self, tstamps=None, method=InterpolationMethod.NEAREST):
        if tstamps is None:
            tstamps = self.ts

        return self._sample_nearest(tstamps)

    def _sample_nearest(self, ts):
        last_idx = len(self._ts) - 1
        idxs = np.searchsorted(self._ts, ts, side="left")
        idxs[idxs > last_idx] = last_idx

        return StreamSampler(self.raw_stream, self.ts[idxs], self.audio_or_video)

    def __iter__(self):
        closest_idxs = np.searchsorted(self.raw_stream.ts, self.ts, side="left")
        for frame_idx in closest_idxs:
            for video_part in self.raw_stream.video_parts:
                if frame_idx < len(video_part.timestamps):
                    frame = video_part.goto_index(frame_idx, self.frame_generators[video_part])
                    actual_ts = video_part.timestamps[frame_idx]

                    yield TimestampedFrame(frame, actual_ts)
                    break
                else:
                    frame_idx -= len(video_part.timestamps)

    @property
    def data(self):
        return self.raw_stream

    @property
    def ts(self):
        return self._ts

    def to_numpy(self):
        if self.audio_or_video == "video":
            return np.array([frame.rgb for frame in self])
        elif self.audio_or_video == "audio":
            return np.array([frame.to_ndarray() for frame in self])

    def __len__(self):
        return len(self._ts)


class AVStreamPart:
    def __init__(self, container, audio_or_video):
        self.container = container
        self.timestamps = None
        self.current_frame = None

        self.audio_or_video = audio_or_video

    def goto_index(self, frame_idx, frame_generator):
        stream = getattr(self.container.streams, self.audio_or_video)[0]

        seek_distance = frame_idx - self.frame_idx
        if seek_distance < 0 or seek_distance > 40:
            self.container.seek(
                int(self._pts[frame_idx]),
                backward=True,
                any_frame=False,
                stream=stream
            )
            self.current_frame = next(frame_generator)

        for _ in range(self.frame_idx, frame_idx):
            self.current_frame = next(frame_generator)

        return self.current_frame

    @property
    def frame_idx(self):
        if self.current_frame is None:
            return -1

        return np.searchsorted(self._pts, self.current_frame.pts)


class BaseAVStream(StreamSampler):
    def __init__(self, ts, audio_or_video):
        super().__init__(self, ts, audio_or_video)

    @property
    def av_containers(self):
        return [v.container for v in self.video_parts]

    @property
    def av_streams(self):
        return [getattr(c.streams, self.audio_or_video)[0] for c in self.av_containers]
