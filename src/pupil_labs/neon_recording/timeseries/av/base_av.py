import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

import pupil_labs.video as plv
from pupil_labs.neon_recording.constants import (
    AV_INDEX_DTYPE,
    AV_INDEX_FIELD_NAME,
    TIMESTAMP_DTYPE,
)
from pupil_labs.neon_recording.timeseries.array_record import (
    Array,
    Record,
    fields,
)
from pupil_labs.neon_recording.timeseries.timeseries import (
    Timeseries,
    TimeseriesProps,
)
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    join_struct_arrays,
)

if TYPE_CHECKING:
    from pupil_labs.neon_recording.neon_recording import NeonRecording

AVTimeseriesKind = Literal["audio", "video"]

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


class BaseAVTimeseries(Timeseries[Array[BaseAVFrame], BaseAVFrame]):
    """Frames from a media container"""

    kind: AVTimeseriesKind
    base_name: str

    def _load_data_from_recording(
        self,
        recording: "NeonRecording",
    ) -> Array[BaseAVFrame]:
        log.debug(f"NeonRecording: Loading video: {self.base_name}.")

        av_files = find_sorted_multipart_files(
            recording._rec_dir, self.base_name, ".mp4"
        )
        parts_ts = []
        video_readers = []
        for av_file, time_file in av_files:
            if self.kind == "video":
                part_ts = Array(time_file, dtype=TIMESTAMP_DTYPE)  # type: ignore
                container_timestamps = (part_ts["time"] - recording.start_time) / 1e9
                reader = plv.Reader(av_file, self.kind, container_timestamps)
                part_ts = part_ts[: len(reader)]
            elif self.kind == "audio":
                reader = plv.Reader(av_file, self.kind)  # type: ignore
                part_ts = (
                    recording.start_time + (reader.container_timestamps * 1e9)  # type: ignore
                ).astype(TIMESTAMP_DTYPE)
            else:
                raise RuntimeError(f"unknown av stream kind: {self.kind}")

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
            f"{self.name.capitalize()}Frame",
            (BaseAVFrame, AVProps),
            {"dtype": data.dtype, "multi_video_reader": av_reader},
        )
        BoundAVFramesClass = type(
            f"{self.name.capitalize()}Frames",
            (Array, AVProps),
            {
                "record_class": BoundAVFrameClass,
                "dtype": data.dtype,
                "multi_video_reader": av_reader,
            },
        )

        return data.view(BoundAVFramesClass)
