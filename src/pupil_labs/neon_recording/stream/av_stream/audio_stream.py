import pupil_labs.video as plv

from ... import structlog
from .base_av_stream import BaseAVStream, BaseAVStreamFrame

log = structlog.get_logger(__name__)


class AudioFrame(BaseAVStreamFrame):
    _frame: plv.AudioFrame


class AudioStream(BaseAVStream, kind="audio"):
    """
    Audio frames from a camera

    Each item is a :class:`.TimestampedFrame`
    """

    @property
    def rate(self):
        return self.av_reader.rate
