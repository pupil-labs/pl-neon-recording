from ... import structlog
from .base_av_stream import BaseAVStream

log = structlog.get_logger(__name__)


class AudioStream(BaseAVStream):
    def __init__(self, recording):
        super().__init__("audio", "Neon Scene Camera v1", recording, "audio")
