import logging

import pupil_labs.video as plv

from .base_av_stream import BaseAVStream, BaseAVStreamFrame

log = logging.getLogger(__name__)


class AudioFrame(BaseAVStreamFrame):
    _frame: plv.AudioFrame


class AudioStream(BaseAVStream, kind="audio"):
    """Audio frames from a camera"""

    @property
    def rate(self):
        return self.av_reader.rate
