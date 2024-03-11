from typing import Tuple, Optional
import pathlib
import numpy as np

import pupil_labs.video as plv

from .stream import Stream

from ..data_utils import load_with_error_check
from ..time_utils import load_and_convert_tstamps

from .. import structlog
log = structlog.get_logger(__name__)

def _load_video(rec_dir: pathlib.Path, start_ts: float, video_name: str) -> Tuple:
    log.debug(f"NeonRecording: Loading video and associated timestamps: {video_name}.")
    
    if not (rec_dir / (video_name + '.mp4')).exists():
        raise FileNotFoundError(f"File not found: {rec_dir / (video_name + '.mp4')}. Please double check the recording download.")

    container = plv.open(rec_dir / (video_name + '.mp4'))

    # use hardware ts
    ts = load_with_error_check(load_and_convert_tstamps, rec_dir / (video_name + '.time'), "Possible error when converting timestamps.")
    # ts = load_with_error_check(load_and_convert_tstamps, rec_dir / (video_name + '.time_aux'), "Possible error when converting timestamps.")
    
    ts_rel = ts - start_ts

    return container, ts, ts_rel


class VideoStream(Stream):
    def interp_data(self, sorted_ts, method="nearest"):
        if method != "nearest":
            raise ValueError("NeonRecording: Video streams only support nearest neighbor interpolation.")
        elif method == "nearest":
            log.warning("NeonRecording: Video streams only use nearest neighbor interpolation.")

            idxs = np.searchsorted(self.ts, sorted_ts)
            return (self._data.frames[int(i)] for i in idxs)
            # return self.data[idxs]
    
    
    def load(self, rec_dir: pathlib.Path, start_ts: float, file_name: Optional[str] = None) -> None:
        if file_name is not None:
            container, ts, ts_rel = _load_video(rec_dir, start_ts, file_name)

            self._data = container.streams.video[0]
            self.data = self._data.frames
            self.ts = ts
            self.ts_rel = ts_rel
        else:
            raise ValueError("Filename must be provided when loading a VideoStream.")
        

    # TODO
    def to_numpy(self):
        pass
