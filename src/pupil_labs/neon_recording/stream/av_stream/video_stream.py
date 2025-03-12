from functools import cached_property

import numpy as np
import numpy.typing as npt

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
    """Video frames from a camera"""

    @property
    def width(self) -> int | None:
        """Width of image in stream"""
        return self.av_reader.width

    @property
    def height(self) -> int | None:
        """Height of image in stream"""
        return self.av_reader.height


class GrayFrame:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    @cached_property
    def bgr(self) -> npt.NDArray[np.uint8]:
        """Image in bgr format"""
        return 128 * np.ones([self.height, self.width, 3], dtype="uint8")

    @cached_property
    def gray(self) -> npt.NDArray[np.uint8]:
        """Image in gray format"""
        return 128 * np.ones([self.height, self.width], dtype="uint8")
