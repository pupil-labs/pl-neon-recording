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
        abs_timestamp: int,
        rel_timestamp: float,
    ):
        super().__init__(av_frame, time, index, source)
        self.abs_timestamp = abs_timestamp
        self.rel_timestamp = rel_timestamp

    @property
    def abs_ts(self) -> int:
        return self.abs_timestamp

    @property
    def rel_ts(self) -> float:
        return self.rel_timestamp


class VideoFrame(plv.VideoFrame):
    def __init__(
        self,
        av_frame: av.video.frame.VideoFrame,
        time: float,
        index: int,
        source: Any,
        abs_timestamp: int,
        rel_timestamp: float,
    ):
        super().__init__(av_frame, time, index, source)
        self.abs_timestamp = abs_timestamp
        self.rel_timestamp = rel_timestamp

    @property
    def abs_ts(self) -> int:
        return self.abs_timestamp

    @property
    def rel_ts(self) -> float:
        return self.rel_timestamp
