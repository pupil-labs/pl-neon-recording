from pathlib import Path
from typing import Iterator, NamedTuple, Optional, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.matching import MatchingMethod, SampledData, sample
from pupil_labs.neon_recording.neon_timeseries import NeonTimeseries
from pupil_labs.neon_recording.utils import (
    find_sorted_multipart_files,
    load_multipart_data_time_pairs,
)
from pupil_labs.video.array_like import ArrayLike


class EyeStateRecord(NamedTuple):
    ts: int
    pupil_diameter_left: float
    eyeball_center_left: npt.NDArray[np.float64]
    optical_axis_left: npt.NDArray[np.float64]
    pupil_diameter_right: float
    eyeball_center_right: npt.NDArray[np.float64]
    optical_axis_right: npt.NDArray[np.float64]

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return np.concatenate([
            [self.ts, self.pupil_diameter_left],
            self.eyeball_center_left,
            self.optical_axis_left,
            [self.pupil_diameter_right],
            self.eyeball_center_right,
            self.optical_axis_right,
        ])


class EyeState(NeonTimeseries[EyeStateRecord]):
    def __init__(self, time_data: ArrayLike[int], eye_state_data: ArrayLike[float]):
        self._time_data = np.array(time_data)
        self._data = np.array(eye_state_data)

    @staticmethod
    def from_native_recording(rec_dir: Path):
        eye_state_files = find_sorted_multipart_files(rec_dir, "eye_state")
        eye_state_data, time_data = load_multipart_data_time_pairs(
            eye_state_files, "<f4", 2
        )
        eye_state_data = eye_state_data.reshape(-1, 14)
        return EyeState(time_data, eye_state_data)

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._time_data

    ts = timestamps

    @property
    def pupil_diameters(self) -> npt.NDArray[np.float64]:
        return self._data[:, [0, 7]]

    @property
    def pupil_diameter_left(self) -> npt.NDArray[np.float64]:
        return self._data[:, 0]

    @property
    def pupil_diameter_right(self) -> npt.NDArray[np.float64]:
        return self._data[:, 7]

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
        elif isinstance(key, slice):
            return EyeState(self._time_data[key], self._data[key])
        else:
            raise TypeError(f"Invalid argument type {type(key)}")

    def __iter__(self) -> Iterator[EyeStateRecord]:
        for i in range(len(self)):
            yield self[i]

    def sample(
        self,
        timestamps: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
    ) -> SampledData:
        return sample(
            timestamps,
            self,
            method=method,
            tolerance=tolerance,
        )

    def interpolate(self, timestamps: ArrayLike[int]) -> "EyeState":
        timestamps = np.array(timestamps)
        interp_data = []

        for key in [
            "pupil_diameter_left",
            "eye_ball_center_left",
            "optical_axis_left",
            "pupil_diameter_right",
            "eye_ball_center_right",
            "optical_axis_right",
        ]:
            data_source = getattr(self, key)
            if data_source.ndim == 1:
                interp_data.append(np.interp(timestamps, self.timestamps, data_source))
            else:
                for dim in range(data_source.shape[1]):
                    interp_dim = np.interp(
                        timestamps, self.timestamps, data_source[:, dim]
                    )
                    interp_data.append(interp_dim)
        interp_data = np.column_stack(interp_data)
        return EyeState(timestamps, interp_data)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._data,
            columns=[
                "pupil_diameter_left",
                "eye_ball_center_left_x",
                "eye_ball_center_left_y",
                "eye_ball_center_left_z",
                "optical_axis_left_x",
                "optical_axis_left_y",
                "optical_axis_left_z",
                "pupil_diameter_right",
                "eye_ball_center_right_x",
                "eye_ball_center_right_y",
                "eye_ball_center_right_z",
                "optical_axis_right_x",
                "optical_axis_right_y",
                "optical_axis_right_z",
            ],
            index=self._time_data,
        )
