from typing import Optional
import numpy as np
import av
import numpy.typing as npt

from .base_av_stream import BaseAVStream, AVStreamPart
from ... import structlog
from ...utils import find_sorted_multipart_files, load_multipart_timestamps
from pupil_labs.video import MultiPartReader


log = structlog.get_logger(__name__)


class VideoStreamPart(AVStreamPart):
    def __init__(self, container, timestamps):
        super().__init__(container, "video")
        self._pts = []
        self.timestamps = timestamps

        packets = container.demux(video=0)
        self._pts = np.array([packet.pts for packet in packets][:-1])
        container.seek(0)


class VideoStream:
    """
    Video frames from a camera

    Each item is a :class:`.TimestampedFrame`
    """

    def __init__(self, name, base_name, recording):
        self.name = name
        self._base_name = base_name
        self.recording = recording

        log.debug(f"NeonRecording: Loading video: {self._base_name}.")

        video_files = find_sorted_multipart_files(
            self.recording._rec_dir, self._base_name, ".mp4"
        )
        self.reader = MultiPartReader([p[0] for p in video_files])
        self._ts = load_multipart_timestamps([p[1] for p in video_files])

    def sample(self, tstamps: Optional[npt.NDArray] = None):
        if tstamps is None:
            tstamps = self._ts

        last_idx = len(self._ts) - 1
        idxs = np.searchsorted(self._ts, tstamps, side="left")
        idxs[idxs > last_idx] = last_idx

        for t, idx in zip(tstamps, idxs):
            frame = self.reader[int(idx)]
            frame.ts = t
            yield frame

    @property
    def width(self):
        return self.reader.width

    @property
    def height(self):
        return self.reader.height

    @property
    def ts(self):
        return self._ts


class GrayFrame:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self._bgr = None
        self._gray = None

    @property
    def bgr(self):
        if self._bgr is None:
            self._bgr = 128 * np.ones([self.height, self.width, 3], dtype="uint8")

        return self._bgr

    @property
    def gray(self):
        if self._gray is None:
            self._gray = 128 * np.ones([self.height, self.width], dtype="uint8")

        return self._gray
