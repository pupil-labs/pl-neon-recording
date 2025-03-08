from typing import TYPE_CHECKING

import numpy as np

from pupil_labs.neon_recording.stream.array_record import Array, Record, proxy
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)

from .. import structlog
from .stream import Stream, StreamProps

log = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from ..neon_recording import NeonRecording


class WornProps(StreamProps):
    worn = proxy[np.float64]("worn")
    "Worn"


class WornRecord(Record, WornProps):
    def keys(self):
        return [x for x in WornProps.__dict__.keys() if not x.startswith("_")]


class WornArray(Array[WornRecord], WornProps):
    record_class = WornRecord


class WornStream(Stream[WornRecord], WornProps):
    """
    Worn data
    """

    data: WornArray

    def __init__(self, recording: "NeonRecording"):
        log.debug("NeonRecording: Loading worn data")

        worn_200hz_file = recording._rec_dir / "worn_200hz.raw"
        time_200hz_file = recording._rec_dir / "gaze_200hz.time"

        file_pairs = []
        if worn_200hz_file.exists() and time_200hz_file.exists():
            log.debug("NeonRecording: Using 200Hz worn data")
            file_pairs.append((worn_200hz_file, time_200hz_file))
        else:
            log.debug("NeonRecording: Using realtime worn data")
            file_pairs = find_sorted_multipart_files(recording._rec_dir, "worn")

        data = load_multipart_data_time_pairs(file_pairs, np.dtype([("worn", "u1")]))
        super().__init__("worn", recording, data.view(WornArray))
