import av
import numpy as np

from ... import structlog
from ...utils import find_sorted_multipart_files, load_multipart_timestamps
from .base_av_stream import AVStreamPart, BaseAVStream

log = structlog.get_logger(__name__)


class AudioStreamPart(AVStreamPart):
    def __init__(self, container, start_ts):
        super().__init__(container, "audio")
        self._pts = []
        self.timestamps = []

        for packet in container.demux(audio=0):
            if packet.size == 0:
                continue
            self._pts.append(packet.pts)
            abs_ts = int(float(packet.pts * packet.time_base)) + start_ts
            self.timestamps.append(abs_ts)

        container.seek(0)


class AudioStream(BaseAVStream):
    """
    Audio frames from a camera

    Each item is a :class:`.TimestampedFrame`
    """

    def __init__(self, recording):
        self.name = "audio"
        self._base_name = "Neon Scene Camera v1"
        self.recording = recording

        log.debug(f"NeonRecording: Loading audio: {self._base_name}.")

        self.video_parts = []

        video_files = find_sorted_multipart_files(
            self.recording._rec_dir, self._base_name, ".mp4"
        )

        timestamps = []
        file_ts = load_multipart_timestamps([video_files[0][1]])
        for video_file, _ in video_files:
            container = av.open(str(video_file))
            part = AudioStreamPart(container, file_ts[0])
            self.video_parts.append(part)
            timestamps += part.timestamps

        self._ts = np.array(timestamps)

        super().__init__(self._ts, "audio")

    @property
    def rate(self):
        return self.video_parts[0].container.streams.audio[0].rate
