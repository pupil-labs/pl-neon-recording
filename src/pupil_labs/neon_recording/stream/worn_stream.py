import numpy as np

from .. import structlog
from .stream import Stream, StreamProps
from ..utils import find_sorted_multipart_files

from pupil_labs.neon_recording.constants import TIMESTAMP_DTYPE
from pupil_labs.neon_recording.stream.array_record import (
    Array,
    Record,
    join_struct_arrays,
    proxy,
)

log = structlog.get_logger(__name__)


class WornProps(StreamProps):
    worn = proxy[np.int64]("worn")
    "Whether the device is worn"


class WornRecord(Record, WornProps):
    def keys(self):
        return [x for x in WornProps.__dict__.keys() if not x.startswith("_")]


class WornArray(Array[WornRecord], WornProps):
    record_class = WornRecord


class WornStream(Stream):
    """
        Worn
    """

    def __init__(self, recording):
        log.debug("NeonRecording: Loading worn data")

        worn_200hz_file = recording._rec_dir / "worn_200hz.raw"
        time_200hz_file = recording._rec_dir / "gaze_200hz.time"

        worn_file_pairs = []
        if worn_200hz_file.exists() and time_200hz_file.exists():
            log.debug("NeonRecording: Using 200Hz worn data")
            worn_file_pairs.append((worn_200hz_file, time_200hz_file))
        else:
            log.debug("NeonRecording: Using realtime worn data")
            worn_file_pairs = find_sorted_multipart_files(recording._rec_dir, "worn")

        time_data = Array([file for _, file in worn_file_pairs], TIMESTAMP_DTYPE)
        worn_data = Array(
            [file for file, _ in worn_file_pairs],
            fallback_dtype=np.dtype([
                ("worn", "uint8"),
            ]),
        )

        data = join_struct_arrays([time_data, worn_data]).view(WornArray)

        super().__init__("worn", recording, data)
