from ... import structlog
from .audio_stream import AudioStream
from .video_stream import VideoStream
from .av_load import _load_av_container

log = structlog.get_logger(__name__)

class AudioVideoStream():
    def __init__(self, name, file_name, recording):
        self.name = name
        self._recording = recording
        self._file_name = file_name
        self._video_stream = None
        self._audio_stream = None

        self._load()

    @property
    def video_stream(self):
        return self._video_stream
    
    @property
    def audio_stream(self):
        return self._audio_stream

    def _load(self):
        log.info(f"NeonRecording: Loading audio-video: {self._file_name}.")

        container, video_ts = _load_av_container(self._recording._rec_dir, self._file_name)

        self._video_stream = VideoStream(
            "video", self._file_name, self._recording, container=container, ts=video_ts
        )
        self._audio_stream = AudioStream("audio", self._file_name, self._recording, container=container, video_ts=video_ts)
