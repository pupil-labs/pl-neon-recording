import logging

from .base_av_stream import AVStreamKind, BaseAVTimeseries

log = logging.getLogger(__name__)

# TODO: This is not used
# class AudioFrame(BaseAVFrame):
#     _frame: plv.AudioFrame


class AudioTimeseries(BaseAVTimeseries):
    """Audio frames stream"""

    kind: AVStreamKind = "video"

    @property
    def rate(self):
        return self.av_reader.rate
