import typing
import pathlib
import numpy as np

import pupil_labs.video as plv

from .stream import Stream

from ..data_utils import load_with_error_check
from ..time_utils import load_and_convert_tstamps

from .. import structlog
log = structlog.get_logger(__name__)

def _load_video(rec_dir: pathlib.Path, video_name: str, start_ts: float) -> typing.Tuple:
    log.debug(f"NeonRecording: Loading video and associated timestamps: {video_name}.")
    
    if not (rec_dir / (video_name + '.mp4')).exists():
        raise FileNotFoundError(f"File not found: {rec_dir / (video_name + '.mp4')}. Please double check the recording download.")

    container = plv.open(rec_dir / (video_name + '.mp4'))

    # use hardware ts
    ts = load_with_error_check(load_and_convert_tstamps, rec_dir / (video_name + '.time'), "Possible error when converting timestamps.")
    # ts = load_with_error_check(load_and_convert_tstamps, rec_dir / (video_name + '.time_aux'), "Possible error when converting timestamps.")
    
    ts_rel = ts - start_ts

    return  container, ts, ts_rel


class VideoStream(Stream):
    def load(self, rec_dir: pathlib.Path, video_name: str, start_ts: float) -> None:
        container, ts, ts_rel = _load_video(rec_dir, video_name, start_ts)

        self._data = container.streams.video[0]
        self.data = self._data.frames
        self.ts = ts
        self.ts_rel = ts_rel

        # because if they use to_numpy without calling sample,
        # then they still get a sensible result: all the data
        self.closest_idxs = np.arange(len(self.ts))


    # TODO(rob) - probably need a different way here
    # def to_numpy(self):
    #     cid = self.data[self.closest_idxs]
    #     return pd.DataFrame(cid).to_numpy()
