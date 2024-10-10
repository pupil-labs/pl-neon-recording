import numpy as np
import av

from .base_av_stream import BaseAVStream, AVStreamPart
from ... import structlog
from ...utils import find_sorted_multipart_files, load_multipart_timestamps


log = structlog.get_logger(__name__)


class VideoStreamPart(AVStreamPart):
    def __init__(self, container, timestamps):
        super().__init__(container, "video")
        self._pts = []
        self.timestamps = timestamps

        packets = container.demux(video=0)
        self._pts = np.array([packet.pts for packet in packets][:-1])
        container.seek(0)


class VideoStream(BaseAVStream):
    """
    Video frames from a camera

    Each item is a :class:`.TimestampedFrame`
    """

    def __init__(self, name, base_name, recording):
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
            part = VideoStreamPart(container, ts)
            self.video_parts.append(part)
            container_start_idx += container.streams.video[0].frames

        super().__init__(self._ts, "video")

    @property
    def width(self):
        return self.video_parts[0].container.streams.video[0].width

    @property
    def height(self):
        return self.video_parts[0].container.streams.video[0].height


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
