from typing import Any

import av
import av.audio.frame
import av.video.frame

import pupil_labs.video as plv


class AudioFrame(plv.AudioFrame):
    def __init__(
        self,
        av_frame: av.audio.frame.AudioFrame,
        time: float,
        index: int,
        source: Any,
        timestamp: int,
    ):
        super().__init__(av_frame, time, index, source)
        self.timestamp = timestamp

    @property
    def ts(self) -> int:
        return self.timestamp


class VideoFrame(plv.VideoFrame):
    def __init__(
        self,
        av_frame: av.video.frame.VideoFrame,
        time: float,
        index: int,
        source: Any,
        timestamp: int,
    ):
        super().__init__(av_frame, time, index, source)
        self.timestamp = timestamp

    @property
    def ts(self) -> int:
        return self.timestamp
