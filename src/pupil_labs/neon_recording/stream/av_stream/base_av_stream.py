import numpy as np

from ... import structlog
from ..stream import InterpolationMethod
from ...utils import find_sorted_multipart_files, load_multipart_timestamps

import av

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

    def sample(self, tstamps, method=InterpolationMethod.NEAREST):
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

                    ts_frame = TimestampedFrame(frame, actual_ts)
                    yield ts_frame
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
            return np.array([frame.bgr for frame in self])
        elif self.audio_or_video == "audio":
            return np.array([frame.to_ndarray() for frame in self])

    def __len__(self):
        return len(self._ts)


class VideoStreamPart:
    def __init__(self, container, timestamps):
        self.container = container
        self.timestamps = timestamps
        self.current_frame = None
        self.frame_idx = -1

    def goto_index(self, frame_idx, frame_generator):
        video = self.container.streams.video[0]

        seek_distance = frame_idx - self.frame_idx
        if seek_distance < 0 or seek_distance > 40:
            target_rel_timestamp = int(frame_idx / video.average_rate)
            self.container.seek(int(target_rel_timestamp * 1e6), backward=True)
            self.current_frame = next(frame_generator)

        for _ in range(self.frame_idx, frame_idx):
            self.current_frame = next(frame_generator)

        self.frame_idx = frame_idx

        return self.current_frame


class BaseAVStream(StreamSampler):
    def __init__(self, name, base_name, recording, audio_or_video):
        self.name = name
        self._base_name = base_name
        self.recording = recording

        log.debug(f"NeonRecording: Loading video: {self._base_name}.")

        self.video_parts = []

        video_files = find_sorted_multipart_files(self.recording._rec_dir, self._base_name, ".mp4")

        self._ts = load_multipart_timestamps([p[1] for p in video_files])
        container_start_idx = 0
        for (video_file, _) in video_files:
            container = av.open(str(video_file))
            ts = self._ts[container_start_idx: container_start_idx + container.streams.video[0].frames]
            self.video_parts.append(VideoStreamPart(container, ts))

            container_start_idx += container.streams.video[0].frames

        super().__init__(self, self._ts, audio_or_video)
