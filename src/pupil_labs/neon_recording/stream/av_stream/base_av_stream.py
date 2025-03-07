from typing import TYPE_CHECKING, Literal

import numpy as np
import plv

from pupil_labs.neon_recording.constants import (
    AV_INDEX_DTYPE,
    AV_INDEX_FIELD_NAME,
    TIMESTAMP_DTYPE,
)
from pupil_labs.neon_recording.stream.array_record import (
    Array,
    Record,
    join_struct_arrays,
)
from pupil_labs.neon_recording.utils import find_sorted_multipart_files

from ... import structlog
from ..stream import Stream

if TYPE_CHECKING:
    from pupil_labs.neon_recording.neon_recording import NeonRecording

AVStreamKind = Literal["audio", "video"]

log = structlog.get_logger(__name__)


class BaseAVStreamFrame(Record):
    """
    Base AVStreamFrame
    """

    multi_video_reader: plv.MultiReader

    @property
    def plv_frame(self):
        return self.multi_video_reader[self[AV_INDEX_FIELD_NAME]]

    def __getattr__(self, name):
        return getattr(self.plv_frame, name)


class BaseAVStream(Stream):
    """
    Frames from a media container

    Each item is a :class:`.TimestampedFrame`
    """

    kind: AVStreamKind

    def __init_subclass__(cls, kind: AVStreamKind):
        cls.kind = kind

    def __init__(
        self,
        name: str,
        base_name: str,
        recording: "NeonRecording",
    ):
        self.name = name
        self._base_name = base_name
        self.recording = recording

        log.debug(f"NeonRecording: Loading video: {self._base_name}.")

        self.video_parts: list[plv.Reader[plv.VideoFrame]] = []
        av_files = find_sorted_multipart_files(
            self.recording._rec_dir, self._base_name, ".mp4"
        )
        parts_ts = []
        video_readers = []
        for av_file, time_file in av_files:
            if self.kind == "video":
                part_ts = Array(time_file, dtype=TIMESTAMP_DTYPE)
                container_timestamps = (part_ts["ts"] - recording.start_ts_ns) / 1e9
                reader = plv.Reader(str(av_file), self.kind, container_timestamps)
                part_ts = part_ts[: len(reader)]
            elif self.kind == "audio":
                reader = plv.Reader(str(av_file), self.kind)
                part_ts = (reader.container_timestamps * 1e9).astype(TIMESTAMP_DTYPE)
            else:
                raise RuntimeError(f"unknown av stream kind: {self.kind}")

            parts_ts.append(part_ts)
            video_readers.append(reader)

        parts_ts = np.concatenate(parts_ts)

        data = join_struct_arrays(
            [
                parts_ts,
                np.arange(len(parts_ts)).view(AV_INDEX_DTYPE),
            ],
        )
        self.av_reader = plv.MultiReader(video_readers)

        BoundAVFrameClass = type(
            f"{self.name.capitalize()}Frame",
            (BaseAVStreamFrame,),
            dict(dtype=data.dtype, multi_video_reader=self.av_reader),
        )
        BoundAVFramesClass = type(
            f"{self.name.capitalize()}Frames",
            (Array,),
            dict(
                record_class=BoundAVFrameClass,
                dtype=data.dtype,
                multi_video_reader=self.av_reader,
            ),
        )

        super().__init__(name, recording, data.view(BoundAVFramesClass))
