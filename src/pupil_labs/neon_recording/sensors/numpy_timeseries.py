from typing import Iterator, Optional, overload

import numpy as np
import numpy.typing as npt

from pupil_labs.matching.matcher import Sensor
from pupil_labs.video.array_like import ArrayLike


class NumpyTimeseries(Sensor):
    def __init__(
        self,
        timestamps: npt.NDArray[np.int32 | np.float32],
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
