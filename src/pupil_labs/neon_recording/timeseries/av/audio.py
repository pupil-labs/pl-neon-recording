import logging

from .base_av import AVTimeseriesKind, BaseAVTimeseries

log = logging.getLogger(__name__)

# TODO: This is not used
# class AudioFrame(BaseAVFrame):
#     _frame: plv.AudioFrame


class AudioTimeseries(BaseAVTimeseries):
    """Audio frames"""

    kind: AVTimeseriesKind = "video"
    base_name: str = "Neon Scene Camera v1"
    name: str = "audio"

    @property
    def rate(self):
        # return self.av_reader.rate
        raise NotImplementedError()
