import numpy as np

import pupil_labs.video as plv

from .base_av_stream import BaseAVStream, BaseAVStreamFrame


class VideoFrame(BaseAVStreamFrame):
    _frame: plv.VideoFrame

    @property
    def bgr(self):
        return self._frame.bgr

    @property
    def gray(self):
        return self._frame.gray


class VideoStream(BaseAVStream, kind="video"):
    """
    Video frames from a camera

    Each item is a :class:`.TimestampedFrame`
    """

    @property
    def width(self):
        return self.av_reader.width

    @property
    def height(self):
        return self.av_reader.height


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
