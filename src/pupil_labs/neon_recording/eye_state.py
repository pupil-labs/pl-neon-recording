from functools import cached_property
from pathlib import Path
from typing import Iterator, NamedTuple, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.neon_recording.arrayrecord import Array
from pupil_labs.neon_recording.neon_timeseries import NeonTimeseries
from pupil_labs.video import ArrayLike


class EyeStateRecord(NamedTuple):
    abs_timestamp: int
    rel_timestamp: float
    pupil_diameter_left: float
    eyeball_center_left: npt.NDArray[np.float64]
    optical_axis_left: npt.NDArray[np.float64]
    pupil_diameter_right: float
    eyeball_center_right: npt.NDArray[np.float64]
    optical_axis_right: npt.NDArray[np.float64]

    eyelid_angle: npt.NDArray[np.float64] | None = None
    eyelid_aperture: npt.NDArray[np.float64] | None = None

    @property
    def abs_ts(self) -> int:
        return self.abs_timestamp

    @property
    def rel_ts(self) -> float:
        return self.rel_timestamp

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return np.concatenate([
            [self.abs_ts, self.pupil_diameter_left],
            self.eyeball_center_left,
            self.optical_axis_left,
            [self.pupil_diameter_right],
            self.eyeball_center_right,
            self.optical_axis_right,
        ])


class EyeState(NeonTimeseries[EyeStateRecord]):
    def __init__(
        self,
        time_data: ArrayLike[int],
        eye_state_data: ArrayLike[float],
        rec_start: int,
    ):
        self._time_data = np.array(time_data)
        self._data = np.array(eye_state_data)
        self._rec_start = rec_start

    @staticmethod
    def from_native_recording(rec_dir: Path, rec_start: int) -> "EyeState":
        eye_state_data = Array(
            rec_dir.glob("eye_state ps*.raw"),
            fallback_dtype=np.dtype([
                ("pupil_diameter_left_mm", "float32"),
                ("eyeball_center_left_x", "float32"),
                ("eyeball_center_left_y", "float32"),
                ("eyeball_center_left_z", "float32"),
                ("optical_axis_left_x", "float32"),
                ("optical_axis_left_y", "float32"),
                ("optical_axis_left_z", "float32"),
                ("pupil_diameter_right_mm", "float32"),
                ("eyeball_center_right_x", "float32"),
                ("eyeball_center_right_y", "float32"),
                ("eyeball_center_right_z", "float32"),
                ("optical_axis_right_x", "float32"),
                ("optical_axis_right_y", "float32"),
                ("optical_axis_right_z", "float32"),
            ]),
        )
        time_data = Array(rec_dir.glob("eye_state ps*.time"), np.int64)
        return EyeState(time_data, eye_state_data, rec_start)

    @property
    def abs_timestamps(self) -> npt.NDArray[np.int64]:
        return self._time_data

    @cached_property
    def rel_timestamps(self) -> npt.NDArray[np.float64]:
        return (self.abs_timestamps - self._rec_start) / 1e9

    @property
    def pupil_diameters(self) -> npt.NDArray[np.float64]:
        return self._data[["pupil_diameter_left_mm", "pupil_diameter_right_mm"]]

    @property
    def pupil_diameter_left(self) -> npt.NDArray[np.float64]:
        return self._data["pupil_diameter_left_mm"]

    @property
    def pupil_diameter_right(self) -> npt.NDArray[np.float64]:
        return self._data["pupil_diameter_right_mm"]

    @property
    def eyeball_center_left(self) -> npt.NDArray[np.float64]:
        return self._data[
            [
                "eyeball_center_left_x",
                "eyeball_center_left_y",
                "eyeball_center_left_z",
            ]
        ]

    @property
    def eyeball_center_right(self) -> npt.NDArray[np.float64]:
        return self._data[
            [
                "eyeball_center_right_x",
                "eyeball_center_right_y",
                "eyeball_center_right_z",
            ]
        ]

    @property
    def optical_axis_left(self) -> npt.NDArray[np.float64]:
        return self._data[
            [
                "optical_axis_left_x",
                "optical_axis_left_y",
                "optical_axis_left_z",
            ]
        ]

    @property
    def optical_axis_right(self) -> npt.NDArray[np.float64]:
        return self._data[
            [
                "optical_axis_right_x",
                "optical_axis_right_y",
                "optical_axis_right_z",
            ]
        ]

    @property
    def eyelid_angle(self) -> npt.NDArray[np.float64]:
        if "eyelid_angle_top_left" not in self._data.dtype.fields:
            return None

        return self._data[
            [
                "eyelid_angle_top_left",
                "eyelid_angle_bottom_left",
                "eyelid_angle_top_right",
                "eyelid_angle_bottom_right",
            ]
        ]

    @property
    def eyelid_aperture(self) -> npt.NDArray[np.float64]:
        if "eyelid_aperture_mm_left" not in self._data.dtype.fields:
            return None

        return self._data[
            [
                "eyelid_aperture_mm_left",
                "eyelid_aperture_mm_right",
            ]
        ]

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
                self.abs_timestamps[key],
                self.rel_timestamps[key],
                self.pupil_diameter_left[key],
                self.eyeball_center_left[key],
                self.optical_axis_left[key],
                self.pupil_diameter_right[key],
                self.eyeball_center_right[key],
                self.optical_axis_right[key],
                None if self.eyelid_angle is None else self.eyelid_angle[key],
                None if self.eyelid_aperture is None else self.eyelid_aperture[key],
            )
            return record
        else:
            return EyeState(self._time_data[key], self._data[key], self._rec_start)

    def __iter__(self) -> Iterator[EyeStateRecord]:
        for i in range(len(self)):
            yield self[i]

    def interpolate(self, timestamps: ArrayLike[int]) -> "EyeState":
        timestamps = np.array(timestamps)
        interp_arr = np.array(
            [
                np.interp(timestamps, self.abs_timestamps, self._data[k])
                for k in self._data.dtype.fields
            ],
            dtype=self._data.dtype,
        )
        return EyeState(timestamps, interp_arr, self._rec_start)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._data, index=self._time_data)
