import numpy as np

from ... import structlog
from .base_av_stream import BaseAVStream


log = structlog.get_logger(__name__)


class VideoStream(BaseAVStream):
    """
    Video frames from a camera

    Each item is a :class:`.TimestampedFrame`
    """

    def __init__(self, name, base_name, recording):
        super().__init__(name, base_name, recording, "video")

        container = self.video_parts[0].container
        self.width = container.streams.video[0].width
        self.height = container.streams.video[0].height


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
