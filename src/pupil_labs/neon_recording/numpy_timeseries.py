from typing import Iterator, Optional, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from pupil_labs.matching import Matcher, MatchingMethod, TimesArray, Timeseries
from pupil_labs.video.array_like import ArrayLike


class NumpyTimeseries(Timeseries):
    def __init__(
        self,
        timestamps: TimesArray,
        data: Optional[npt.NDArray] = None,
    ):
        self.timestamps = timestamps
        if data is None:
            self.data = timestamps
        else:
            assert len(data) == len(timestamps)
            self.data = data

    @overload
    def __getitem__(self, key: int, /) -> int: ...
    @overload
    def __getitem__(self, key: slice, /) -> ArrayLike[int]: ...
    def __getitem__(self, key: int | slice, /) -> int | ArrayLike[int]:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.timestamps)

    def __iter__(self) -> Iterator[int]:
        for i in range(len(self)):
            yield self[i]

    def sample(
        self, target_ts: TimesArray, method: MatchingMethod = MatchingMethod.NEAREST
    ) -> ArrayLike:
        return Matcher(target_ts, self, method)

    @staticmethod
    def from_dataframe(df: pd.DataFrame, ts_column: str) -> "NumpyTimeseries":
        time: TimesArray = df[ts_column].values.astype(np.float64)
        data = df.drop(columns=[ts_column]).values
        return NumpyTimeseries(time, data)
