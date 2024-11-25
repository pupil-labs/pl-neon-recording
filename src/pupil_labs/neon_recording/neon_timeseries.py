from typing import Optional, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from pupil_labs.matching import MatchingMethod, SampledData
from pupil_labs.video import ArrayLike, Indexer

T = TypeVar("T", covariant=True)


class NeonTimeseries(ArrayLike[T], Protocol[T]):
    @property
    def timestamps(self) -> npt.NDArray[np.int64]: ...

    def sample(
        self,
        timestamps: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
    ) -> SampledData[T]: ...

    @property
    def by_abs_timestamp(self) -> Indexer[T]: ...

    @property
    def by_rel_timestamp(self) -> Indexer[T]: ...
