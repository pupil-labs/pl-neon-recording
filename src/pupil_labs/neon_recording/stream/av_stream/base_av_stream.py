import logging
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

import pupil_labs.video as plv
from pupil_labs.neon_recording.constants import (
    AV_INDEX_DTYPE,
    AV_INDEX_FIELD_NAME,
    TIMESTAMP_DTYPE,
)
from pupil_labs.neon_recording.sample import match_ts
from pupil_labs.neon_recording.stream.array_record import Array, Record, fields

# from ..stream import Stream, StreamProps
from pupil_labs.neon_recording.timeseries import Timeseries, TimeseriesProps
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    join_struct_arrays,
)
from pupil_labs.video import ArrayLike

if TYPE_CHECKING:
    from pupil_labs.neon_recording.neon_recording import NeonRecording

AVStreamKind = Literal["audio", "video"]

log = logging.getLogger(__name__)


class AVProps(TimeseriesProps):
    idx = fields[np.int32]("idx")
    "Frame index in stream"


class BaseAVFrame(Record):
    multi_video_reader: plv.MultiReader

    @property
    def plv_frame(self):
        return self.multi_video_reader[self[AV_INDEX_FIELD_NAME]]

    def __getattr__(self, name):
        return getattr(self.plv_frame, name)


T = TypeVar("T", bound="BaseAVTimeseries")


class BaseAVTimeseries(Timeseries[Array[BaseAVFrame], BaseAVFrame]):
    """Frames from a media container"""

    kind: AVStreamKind

    # def __init_subclass__(cls, kind: AVStreamKind):
    #     cls.kind = kind

    def __init__(
        self,
        data: Array[BaseAVFrame],
        name: str,
        recording: "NeonRecording",
        av_reader: plv.MultiReader,
        # kind: AVStreamKind = "video",
    ):
        super().__init__(data, name, recording)
        self.av_reader = av_reader
        # self.kind = kind

    @classmethod
    def from_recording(
        cls: type[T],
        name: str,
        base_name: str,
        recording: "NeonRecording",
        # kind: AVStreamKind,
    ) -> T:
        log.debug(f"NeonRecording: Loading video: {base_name}.")

        av_files = find_sorted_multipart_files(recording._rec_dir, base_name, ".mp4")
        parts_ts = []
        video_readers = []
        for av_file, time_file in av_files:
            if cls.kind == "video":
                part_ts = Array(time_file, dtype=TIMESTAMP_DTYPE)  # type: ignore
                container_timestamps = (part_ts["ts"] - recording.start_ts) / 1e9
                reader = plv.Reader(av_file, cls.kind, container_timestamps)
                part_ts = part_ts[: len(reader)]
            elif cls.kind == "audio":
                reader = plv.Reader(av_file, cls.kind)  # type: ignore
                part_ts = (
                    recording.start_ts + (reader.container_timestamps * 1e9)  # type: ignore
                ).astype(TIMESTAMP_DTYPE)
            else:
                raise RuntimeError(f"unknown av stream kind: {cls.kind}")

            parts_ts.append(part_ts)
            video_readers.append(reader)

        parts_ts = np.concatenate(parts_ts)
        idxs = np.empty(len(parts_ts), dtype=AV_INDEX_DTYPE)
        idxs[AV_INDEX_FIELD_NAME] = np.arange(len(parts_ts))

        data = join_struct_arrays(
            [
                parts_ts,  # type: ignore
                idxs,
            ],
        )
        av_reader = plv.MultiReader(video_readers)

        BoundAVFrameClass = type(
            f"{name.capitalize()}Frame",
            (BaseAVFrame, AVProps),
            {"dtype": data.dtype, "multi_video_reader": av_reader},
        )
        BoundAVFramesClass = type(
            f"{name.capitalize()}Frames",
            (Array, AVProps),
            {
                "record_class": BoundAVFrameClass,
                "dtype": data.dtype,
                "multi_video_reader": av_reader,
            },
        )

        return cls(
            data.view(BoundAVFramesClass),
            name,
            recording,
            av_reader,
            # kind,
        )

    def sample(
        self: T,
        target_ts: ArrayLike[int],
        method: Literal["nearest", "before", "after"] = "nearest",
        tolerance: int | None = None,
    ) -> T:
        indices = match_ts(target_ts, self.ts, method, tolerance)
        return self.__class__(
            self._data[indices],  # type: ignore
            self.name,
            self.recording,
            self.av_reader,
            # self.kind,
        )
