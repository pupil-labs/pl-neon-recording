import pathlib
from enum import Enum
from typing import Optional, Tuple

import pupil_labs.video as plv

from .. import structlog
from ..data_utils import load_with_error_check
from ..time_utils import load_and_convert_tstamps
from .stream import Stream

log = structlog.get_logger(__name__)


def _load_video(rec_dir: pathlib.Path, start_ts: float, video_name: str) -> Tuple:
    log.debug(f"NeonRecording: Loading video and associated timestamps: {video_name}.")

    if not (rec_dir / (video_name + ".mp4")).exists():
        raise FileNotFoundError(
            f"File not found: {rec_dir / (video_name + '.mp4')}. Please double check the recording download."
        )

    container = plv.open(rec_dir / (video_name + ".mp4"))

    # use hardware ts
    ts = load_with_error_check(
        load_and_convert_tstamps,
        rec_dir / (video_name + ".time"),
        "Possible error when converting timestamps.",
    )
    # ts = load_with_error_check(load_and_convert_tstamps, rec_dir / (video_name + '.time_aux'), "Possible error when converting timestamps.")

    ts_rel = ts - start_ts

    return container, ts, ts_rel


class VideoType(Enum):
    EYE = 1
    SCENE = 2


class VideoStream(Stream):
    def __init__(self, name, type):
        super().__init__(name)
        self.type = type

    def linear_interp(self, sorted_ts):
        raise NotImplementedError(
            "NeonRecording: Video streams only support nearest neighbor interpolation."
        )

    def load(
        self, rec_dir: pathlib.Path, start_ts: float, file_name: Optional[str] = None
    ) -> None:
        if file_name is not None:
            container, ts, ts_rel = _load_video(rec_dir, start_ts, file_name)

            self._backing_data = container.streams.video[0]
            self.data = self._backing_data.frames
            self.ts = ts
            self.ts_rel = ts_rel
        else:
            raise ValueError("Filename must be provided when loading a VideoStream.")
