from ... import structlog
from .base_stream import BaseStream

log = structlog.get_logger(__name__)


class AudioStream(BaseStream):
    def __init__(self, recording):
        super().__init__("audio", "Neon Scene Camera v1", recording, "audio")
