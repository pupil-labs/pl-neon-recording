from typing import Optional, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from pupil_labs.matching import MatchingMethod, SampledData
from pupil_labs.video import ArrayLike

T = TypeVar("T", covariant=False)


class NeonTimeseries(ArrayLike[T], Protocol):
    @property
    def timestamps(self) -> npt.NDArray[np.int64]: ...

    def sample(
        self,
        timestamps: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[float] = None,
    ) -> SampledData[T]: ...
