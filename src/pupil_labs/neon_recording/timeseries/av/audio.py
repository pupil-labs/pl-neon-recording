import logging
from functools import cached_property

from .base_av import AVTimeseriesKind, BaseAVTimeseries

log = logging.getLogger(__name__)

# TODO: This is not used
# class AudioFrame(BaseAVFrame):
#     _frame: plv.AudioFrame


class AudioTimeseries(BaseAVTimeseries):
    """Audio frames"""

    kind: AVTimeseriesKind = "audio"
    base_name: str = "Neon Scene Camera v1"
    name: str = "audio"

    @cached_property
    def rate(self):
        return self._data[0].multi_video_reader.rate
