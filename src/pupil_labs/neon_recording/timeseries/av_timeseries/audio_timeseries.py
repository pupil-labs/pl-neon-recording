import logging

from .base_av_timeseries import AVTimeseriesKind, BaseAVTimeseries

log = logging.getLogger(__name__)

# TODO: This is not used
# class AudioFrame(BaseAVFrame):
#     _frame: plv.AudioFrame


class AudioTimeseries(BaseAVTimeseries):
    """Audio frames"""

    kind: AVTimeseriesKind = "video"

    @property
    def rate(self):
        return self.av_reader.rate
