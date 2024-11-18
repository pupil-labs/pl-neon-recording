from dataclasses import dataclass

import pupil_labs.video as plv


@dataclass(frozen=True)
class AudioFrame(plv.AudioFrame):
    timestamp: int


@dataclass(frozen=True)
class VideoFrame(plv.VideoFrame):
    timestamp: int
