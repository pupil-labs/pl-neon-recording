import numpy as np
import numpy.typing as npt

from pupil_labs.neon_recording.neon_timeseries import NeonTimeseries
from pupil_labs.video.frame_slice import LAZY_FRAME_SLICE_LIMIT
from pupil_labs.video.frame_slice import FrameSlice as plv_FrameSlice
from pupil_labs.video.frame_slice import FrameType as plv_FrameType


class FrameSlice(plv_FrameSlice[plv_FrameType]):
    target: NeonTimeseries[plv_FrameType]

    def __init__(
        self,
        target: NeonTimeseries[plv_FrameType],
        slice_value: slice,
        lazy_frame_slice_limit: int = LAZY_FRAME_SLICE_LIMIT,
    ):
        super().__init__(target, slice_value, lazy_frame_slice_limit)

    @property
    def abs_timestamps(self) -> npt.NDArray[np.int64]:
        return self.target.abs_timestamps[self.slice]

    @property
    def abs_ts(self) -> npt.NDArray[np.int64]:
        return self.target.abs_ts[self.slice]

    @property
    def rel_timestamps(self) -> npt.NDArray[np.float64]:
        return self.target.rel_timestamps[self.slice]

    @property
    def rel_ts(self) -> npt.NDArray[np.float64]:
        return self.target.rel_ts[self.slice]
