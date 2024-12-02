from functools import cached_property
from typing import Optional, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from pupil_labs.matching import MatchingMethod, SampledData
from pupil_labs.video import ArrayLike, Indexer

T = TypeVar("T", covariant=True)


class NeonTimeseries(ArrayLike[T], Protocol[T]):
    @property
    def abs_timestamp(self) -> npt.NDArray[np.int64]:
        """Absolute timestamps in nanoseconds since epoch"""
        ...

    @property
    def abs_ts(self) -> npt.NDArray[np.int64]:
        """Alias for abs_timestamp"""
        ...

    def sample(
        self,
        timestamps: ArrayLike[int],
        method: MatchingMethod = MatchingMethod.NEAREST,
        tolerance: Optional[int] = None,
    ) -> SampledData[T]:
        """Match the data to the given timestamps.

        Args:
            timestamps: The timestamps to match the data to.
            method: The method to use for matching.
            tolerance: The maximum time difference in nanoseconds for matching.

        """
        ...

    @cached_property
    def rel_timestamp(self) -> npt.NDArray[np.float64]:
        """Relative timestamps in seconds in relation to the recording beginning."""
        ...

    @property
    def rel_ts(self) -> npt.NDArray[np.float64]:
        """Alias for rel_timestamp"""
        ...

    @property
    def by_abs_timestamp(self) -> Indexer[T]:
        """Time-based access to video frames using absolute timestamps."""
        ...

    @property
    def by_rel_timestamp(self) -> Indexer[T]:
        """Time-based access to video frames using relative timestamps.

        Timestamps are relative to the beginning of the recording.
        """
        ...
