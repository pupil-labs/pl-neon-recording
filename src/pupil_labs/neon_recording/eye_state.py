from pathlib import Path
from typing import Iterator, NamedTuple, Optional, overload

import numpy as np
import numpy.typing as npt

from pupil_labs.matching import Matcher, MatchingMethod
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)


class EyeStateRecord(NamedTuple):
    ts: int
    pupil_diameter_left: npt.NDArray[np.float64]
    eyeball_center_left: npt.NDArray[np.float64]
    optical_axis_left: npt.NDArray[np.float64]
    pupil_diameter_right: float
    eyeball_center_right: npt.NDArray[np.float64]
    optical_axis_right: npt.NDArray[np.float64]


class EyeState:
    def __init__(self, rec_dir: Path):
        eye_state_files = find_sorted_multipart_files(rec_dir, "eye_state")
        eye_state_data, time_data = load_multipart_data_time_pairs(
            eye_state_files, "<f4", 2
        )
        self._time_data = time_data
        self._data = eye_state_data.reshape(-1, 14)

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._time_data

    ts = timestamps

    @property
    def pupil_diameters(self) -> npt.NDArray[np.float64]:
        return self._data[:, [0, 7]]

    @property
    def eye_ball_center_left(self) -> npt.NDArray[np.float64]:
        return self._data[:, [1, 2, 3]]

    @property
    def eye_ball_center_right(self) -> npt.NDArray[np.float64]:
        return self._data[:, [8, 9, 10]]

    @property
    def optical_axis_left(self) -> npt.NDArray[np.float64]:
        return self._data[:, [4, 5, 6]]

    @property
    def optical_axis_right(self) -> npt.NDArray[np.float64]:
        return self._data[:, [11, 12, 13]]

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return self._data

    def __len__(self) -> int:
        return len(self._time_data)

    @overload
    def __getitem__(self, key: int, /) -> EyeStateRecord: ...
    @overload
    def __getitem__(self, key: slice, /) -> "EyeState": ...
    def __getitem__(self, key: int | slice) -> "EyeStateRecord | EyeState":
        if isinstance(key, int):
            record = EyeStateRecord(
                self._time_data[key],
                self._data[key, 0],
                self._data[key, 1:4],
                self._data[key, 4:7],
                self._data[key, 7],
                self._data[key, 8:11],
                self._data[key, 11:14],
            )
            return record
        else:
            # TODO
            raise NotImplementedError
        return record

    def __iter__(self) -> Iterator[EyeStateRecord]:
        for i in range(len(self)):
            yield self[i]

    def sample(
        self,
        timestamps: npt.NDArray[np.float64],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[float] = None,
        include_timeseries_ts: bool = False,
        include_target_ts: bool = False,
    ) -> Matcher:
        return Matcher(
            timestamps,
            self,
            method=method,
            tolerance=tolerance,
            include_timeseries_ts=include_timeseries_ts,
            include_target_ts=include_target_ts,
        )
